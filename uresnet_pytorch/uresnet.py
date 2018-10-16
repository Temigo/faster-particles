from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

from faster_particles.demo_ppn import get_data
from faster_particles.config import PPNConfig
from faster_particles.display_utils import display_uresnet, extract_voxels

import numpy as np
import os
import time
import glob
import re

# Normal UResNet and sparse version
from uresnet_pytorch.base_uresnet import UResNet
from uresnet_pytorch.sparse_uresnet import UResNet as UResNetSparse
#
# from sparsio.iotools import io_factory

# Accelerate *if all input sizes are same*
torch.backends.cudnn.benchmark = True

# def dataloader(i):
#     print("loading data ", i)
#     i = i % num_img
#     data = f.root.data[i].reshape((batch_size, 1, N, N, N))
#     label = f.root.label[i].astype(int).reshape((batch_size, N, N, N))
#     return torch.from_numpy(data).cuda(), torch.from_numpy(label).cuda()


# def get_data_sparse(cfg):
#     flags = {
#         'IO_TYPE': 'larcv',
#         'INPUT_FILE': cfg.TEST_DATA,
#         'OUTPUT_FILE': '',
#         'BATCH_SIZE': cfg.BATCH_SIZE,
#         'DATA_KEY': 'data',
#         'LABEL_KEY': '',
#         'SHUFFLE': True,
#     }
#     io = io_factory(flags)
#     io.initialize()
#     num_entries = io.num_entries()
#     i = 0
#     while i < num_entries:
#         data, label, idx = io.next()
#         msg = str(i) + '/' + str(num_entries) + ' ... '  + str(idx) + ' ' + str(data[0].shape)
#         if label:
#             msg += str(label[0].shape)
#         print(msg)
#         i += len(data)
#     io.finalize()


def train_demo(cfg, net, criterion, optimizer, lr_scheduler, is_training):
    # Data generator
    train_data, test_data = get_data(cfg)
    if is_training:
        data = train_data
    else:
        data = test_data

    # Initialize the network the right way
    # net.train and net.eval account for differences in dropout/batch norm
    # during training and testing
    start = 0
    if is_training:
        net.train().cuda()
    else:
        net.eval().cuda()
    if cfg.WEIGHTS_FILE_BASE is not None and cfg.WEIGHTS_FILE_BASE != '':
        print('Restoring weights from %s...' % cfg.WEIGHTS_FILE_BASE)
        with open(cfg.WEIGHTS_FILE_BASE, 'rb') as f:
            checkpoint = torch.load(f)
            net.load_state_dict(checkpoint['state_dict'])
            # print(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start = checkpoint['epoch'] + 1
        print('Done.')
    print('Done.')

    metrics = {'acc_all': [], 'acc_nonzero': [], 'loss': [], 'memory': [],
               'durations_forward': [], 'durations_backward': [],
               'durations_cuda': [], 'durations': [], 'durations_data': []}
    # Only enable gradients if we are training
    # with torch.set_grad_enabled(is_training):
    for i in range(cfg.MAX_STEPS):  # use with torch.no_grad() for test network
        print("Step %d/%d" % (i, cfg.MAX_STEPS))
        start_step = time.time()
        start = time.time()
        blob = data.forward()
        end = time.time()
        metrics['durations_data'].append(end-start)
        if cfg.SPARSE:
            start = time.time()
            coords = torch.from_numpy(blob['voxels']).cuda()
            features = torch.from_numpy(np.reshape(blob['voxels_value'], (-1, 1))).cuda()
            end = time.time()
            metrics['durations_cuda'].append(end-start)
            start = time.time()
            predictions_raw = net(coords, features)  # size N_voxels x num_classes
            end = time.time()
            metrics['durations_forward'].append(end-start)
            label_voxels, labels = extract_voxels(blob['labels'])
            labels = torch.from_numpy(labels).cuda().type(torch.cuda.LongTensor)
        else:
            start = time.time()
            image = torch.from_numpy(np.moveaxis(blob['data'], -1, 1)).cuda()
            labels = torch.from_numpy(blob['labels']).cuda().type(torch.cuda.LongTensor)
            end = time.time()
            metrics['durations_cuda'].append(end-start)
            start = time.time()
            predictions_raw = net(image)
            end = time.time()
            metrics['durations_forward'].append(end-start)

        loss = criterion(predictions_raw, labels)
        if is_training:
            start = time.time()
            # lr_scheduler.step()  # Decay learning rate
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients of all variables wrt loss
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)  # Clip gradient
            optimizer.step()  # update using computed gradients
            end = time.time()
            metrics['durations_backward'].append(end-start)
        metrics['loss'].append(loss.item())
        print("\tLoss = ", metrics['loss'][-1])

        # Accuracy
        predicted_labels = torch.argmax(predictions_raw, dim=1)
        acc_all = (predicted_labels == labels).sum().item() / float(labels.numel())
        nonzero_px = labels > 0
        nonzero_prediction = predicted_labels[nonzero_px]
        nonzero_label = labels[nonzero_px]
        acc_nonzero = (nonzero_prediction == nonzero_label).sum().item() / float(nonzero_label.numel())
        metrics['acc_all'].append(acc_all)
        metrics['acc_nonzero'].append(acc_nonzero)
        print("\tAccuracy = ", metrics['acc_all'][-1], " - Nonzero accuracy = ", metrics['acc_nonzero'][-1])

        metrics['memory'].append(torch.cuda.memory_allocated())

        if is_training and i % 100 == 0:
            for attr in metrics:
                np.savetxt(os.path.join(cfg.OUTPUT_DIR, '%s_%d.csv' % (attr, i)), metrics[attr], delimiter=',')

        if is_training and i % 100 == 0:
            filename = os.path.join(cfg.OUTPUT_DIR, 'model-%d.ckpt' % i)
            # with open(filename, 'wb'):
            torch.save({
                'epoch': i,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, filename)

        if not is_training:
            print('Display...')
            if cfg.SPARSE:
                final_predictions = np.zeros((1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
                indices = label_voxels.T
                final_predictions[0, indices[0], indices[1], indices[2]] = predicted_labels.cpu().data.numpy()
                display_uresnet(blob, cfg,
                                index=i,
                                predictions=final_predictions)
            else:
                display_uresnet(blob, cfg,
                                index=i,
                                predictions=predicted_labels.cpu().data.numpy())
            print('Done.')
        end_step = time.time()
        metrics['durations'].append(end_step - start_step)
    print("Average duration = %f s" % metrics['durations'].mean())


def test(cfg, net):
    # data = get_data_sparse(cfg)
    _, data = get_data(cfg)

    # Initialize the network the right way
    # net.train and net.eval account for differences in dropout/batch norm
    # during training and testing
    net.eval().cuda()
    metrics = {'acc_all': [], 'acc_nonzero': [], 'loss': []}
    metrics_mean = {'acc_all': [], 'acc_nonzero': [], 'loss': []}
    metrics_std = {'acc_all': [], 'acc_nonzero': [], 'loss': []}
    durations_mean = {'cuda': [], 'loss': [], 'forward': [], 'acc': []}
    durations_std = {'cuda': [], 'loss': [], 'forward': [], 'acc': []}
    durations, durations_cuda, durations_loss, durations_acc = [], [], [], []
    steps = []
    print('Listing weights...')
    weights = glob.glob(os.path.join(cfg.WEIGHTS_FILE_BASE, "*.ckpt"))
    weights.sort()
    print('Done.')

    blobs = []
    print('Fetch data...')
    for i in range(cfg.MAX_STEPS):
        print("%d/%d" % (i, cfg.MAX_STEPS))
        blob = data.forward()
        blob.pop('data')
        blob['label_voxels'], blob['label_values'] = extract_voxels(blob['labels'])
        blob.pop('labels')
        blobs.append(blob)
    print('Done.')

    for w in weights:
        step = int(re.findall(r'model-(\d+)', w)[0])
        steps.append(step)
        print('Restoring weights from %s...' % w)
        with open(w, 'rb') as f:
            checkpoint = torch.load(f)
            net.load_state_dict(checkpoint['state_dict'])
        print('Done.')
        for i, blob in enumerate(blobs):  # FIXME
            print("Step %d/%d" % (i, cfg.MAX_STEPS))
            if cfg.SPARSE:
                start = time.time()
                coords = torch.from_numpy(blob['voxels']).cuda()
                features = torch.from_numpy(np.reshape(blob['voxels_value'], (-1, 1))).cuda()
                label_voxels, labels = blob['label_voxels'], blob['label_values']
                labels = torch.from_numpy(labels).cuda().type(torch.cuda.LongTensor)
                end = time.time()
                durations_cuda.append(end-start)

                start = time.time()
                predictions_raw = net(coords, features)  # size N_voxels x num_classes
                end = time.time()
                durations.append(end-start)

            else:
                start = time.time()
                image = torch.from_numpy(np.moveaxis(blob['data'], -1, 1)).cuda()
                labels = torch.from_numpy(blob['labels']).cuda().type(torch.cuda.LongTensor)
                end = time.time()
                durations_cuda.append(end - start)

                start = time.time()
                predictions_raw = net(image)
                end = time.time()
                durations.append(end-start)

            start = time.time()
            loss = criterion(predictions_raw, labels)
            end = time.time()
            durations_loss.append(end - start)
            metrics['loss'].append(loss.item())
            print("\tLoss = ", metrics['loss'][-1])

            # Accuracy
            start = time.time()
            predicted_labels = torch.argmax(predictions_raw, dim=1)
            acc_all = (predicted_labels == labels).sum().item() / float(labels.numel())
            nonzero_px = labels > 0
            nonzero_prediction = predicted_labels[nonzero_px]
            nonzero_label = labels[nonzero_px]
            acc_nonzero = (nonzero_prediction == nonzero_label).sum().item() / float(nonzero_label.numel())
            end = time.time()
            durations_acc.append(end - start)
            metrics['acc_all'].append(acc_all)
            metrics['acc_nonzero'].append(acc_nonzero)
            print("\tAccuracy = ", metrics['acc_all'][-1], " - Nonzero accuracy = ", metrics['acc_nonzero'][-1])

        metrics_mean['loss'].append(np.array(metrics['loss']).mean())
        metrics_std['loss'].append(np.array(metrics['loss']).std())
        metrics_mean['acc_all'].append(np.array(metrics['acc_all']).mean())
        metrics_std['acc_all'].append(np.array(metrics['acc_all']).std())
        metrics_mean['acc_nonzero'].append(np.array(metrics['acc_nonzero']).mean())
        metrics_std['acc_nonzero'].append(np.array(metrics['acc_nonzero']).std())
        durations_mean['cuda'].append(np.array(durations_cuda).mean())
        durations_std['cuda'].append(np.array(durations_cuda).std())
        durations_mean['loss'].append(np.array(durations_loss).mean())
        durations_std['loss'].append(np.array(durations_loss).std())
        durations_mean['forward'].append(np.array(durations).mean())
        durations_std['forward'].append(np.array(durations).std())
        durations_mean['acc'].append(np.array(durations_acc).mean())
        durations_std['acc'].append(np.array(durations_acc).std())
        durations, durations_cuda, durations_loss, durations_acc = [], [], [], []
        metrics = {'acc_all': [], 'acc_nonzero': [], 'loss': []}

        print('Mean cuda duration = %f s' % durations_mean['cuda'][-1])
        print('Mean loss duration = %f s' % durations_mean['loss'][-1])
        print('Mean acc duration = %f s' % durations_mean['acc'][-1])
        print('Mean forward duration = %f s' % durations_mean['forward'][-1])

        print('Mean acc = %f s' % metrics_mean['acc_nonzero'][-1])

        np.savetxt(os.path.join(cfg.OUTPUT_DIR, 'steps_%d.csv' % step), steps, delimiter=',')
        for attr in metrics:
            np.savetxt(os.path.join(cfg.OUTPUT_DIR, '%s_mean_%d.csv' % (attr, step)), metrics_mean[attr], delimiter=',')
            np.savetxt(os.path.join(cfg.OUTPUT_DIR, '%s_std_%d.csv' % (attr, step)), metrics_std[attr], delimiter=',')
        for attr in durations_mean:
            np.savetxt(os.path.join(cfg.OUTPUT_DIR, 'durations_%s_mean_%d.csv' % (attr, step)), durations_mean[attr], delimiter=',')
            np.savetxt(os.path.join(cfg.OUTPUT_DIR, 'durations_%s_std_%d.csv' % (attr, step)), durations_std[attr], delimiter=',')


if __name__ == '__main__':
    # Retrieve dataset with LarcvGenerator
    cfgargs = {
        'DISPLAY_DIR': 'display/sparse11',
        'OUTPUT_DIR': '/data/sparse11',
        'LOG_DIR': 'log/sparse11',  # useless
        'BASE_NET': 'uresnet',
        'NET': 'base',
        'MAX_STEPS': 10,
        'DATA_3D': True,
        'SPARSE': True,
        'DATA': "/data/dlprod_ppn_v08_p02_filtered/train_p02.root",
        'TEST_DATA': "/data/dlprod_ppn_v08_p02_filtered/test_p02.root",
        # 'WEIGHTS_FILE_BASE': '/data/sparse10',
        'IMAGE_SIZE': 192,
        'BATCH_SIZE': 32,
        'GPU': '1',
        'NUM_STRIDES': 5,
        'BASE_NUM_OUTPUTS': 16,
    }
    cfg = PPNConfig(**cfgargs)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    if not os.path.isdir(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR)
    if not os.path.isdir(cfg.DISPLAY_DIR):
        os.makedirs(cfg.DISPLAY_DIR)

    is_training = True

    # Instantiate and move to GPU the network
    print('Building network...')
    if cfg.SPARSE:
        net = UResNetSparse(True,
                      num_strides=cfg.NUM_STRIDES,
                      base_num_outputs=cfg.BASE_NUM_OUTPUTS)
    else:
        net = UResNet(True,
                      num_strides=cfg.NUM_STRIDES,
                      base_num_outputs=cfg.BASE_NUM_OUTPUTS)
    # print(net)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    logsoftmax = nn.LogSoftmax(dim=1)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 1000, gamma=0.1)

    train_demo(cfg, net, criterion, optimizer, lr_scheduler, is_training)
    # test(cfg, net)
    # get_data_sparse(cfg)
