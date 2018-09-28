# *-* encoding: utf-8 *-*
# Demo for PPN

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
import os
import glob
import time
from sklearn.cluster import DBSCAN

from faster_particles.display_utils import display, display_uresnet, \
                                            display_ppn_uresnet, display_blob
from faster_particles.ppn import PPN
from faster_particles.base_net.uresnet import UResNet
from faster_particles.base_net import basenets
from faster_particles.metrics import PPNMetrics, UResNetMetrics
from faster_particles.data import ToydataGenerator, LarcvGenerator, \
                                HDF5Generator, CSVGenerator
from faster_particles.cropping import cropping_algorithms
from faster_particles.display_utils import extract_voxels


def get_data(cfg):
    """
    Define data generators (toydata or LArCV)
    """
    if cfg.TEST_DATA == "":
        cfg.TEST_DATA = cfg.DATA
    if cfg.DATA_TYPE == 'toydata':
        train_data = ToydataGenerator(cfg)
        test_data = ToydataGenerator(cfg)
    elif cfg.DATA_TYPE == 'hdf5':
        train_data = HDF5Generator(cfg, filelist=cfg.DATA)
        test_data = HDF5Generator(cfg, filelist=cfg.TEST_DATA, is_testing=True)
    elif cfg.DATA_TYPE == 'csv':
        train_data = CSVGenerator(cfg, filelist=cfg.DATA)
        test_data = CSVGenerator(cfg, filelist=cfg.TEST_DATA)
    else:  # default is LArCV data
        train_data = LarcvGenerator(cfg, ioname="train",
                                    filelist=get_filelist(cfg.DATA))
        test_data = LarcvGenerator(cfg, ioname="test",
                                   filelist=get_filelist(cfg.TEST_DATA))
    return train_data, test_data


def load_weights(cfg, sess):
    """
    Restore TF weights stored in checkpoint file.
    """
    print("Restoring checkpoint file...")
    scopes = []
    if cfg.WEIGHTS_FILE_PPN is not None:
        scopes.append((lambda x: 'small_uresnet' not in x, cfg.WEIGHTS_FILE_PPN))
    # Restore variables for base net if given checkpoint file
    elif cfg.WEIGHTS_FILE_BASE is not None:
        if cfg.NET in ['ppn', 'ppn_ext', 'full']: # load only relevant layers of base network
		    scopes.append((lambda x: cfg.BASE_NET in x and "optimizer" not in x, cfg.WEIGHTS_FILE_BASE))
            #scopes.append((lambda x: cfg.BASE_NET in x, cfg.WEIGHTS_FILE_BASE))
        else: # load for full base network
            scopes.append((lambda x: cfg.BASE_NET in x, cfg.WEIGHTS_FILE_BASE))
    if cfg.WEIGHTS_FILE_SMALL is not None:
        scopes.append((lambda x: "small_uresnet" in x, cfg.WEIGHTS_FILE_SMALL))

    for scope, weights_file in scopes:
        print('Restoring %s...' % weights_file)
        variables_to_restore = [v for v in tf.global_variables() if scope(v.name)]
        print("- ignoring %d/%d variables" % (len(tf.global_variables()) - len(variables_to_restore), len(tf.global_variables())))
        if len(variables_to_restore) > 0:
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, weights_file)
        else:
            print("WARNING No variable was restored from weights file %s." % weights_file)
    print("Done.")


def get_filelist(ls_command):
    """
    Returns a list of files as a string *without spaces*
    e.g. "["file1.root","file2.root"]"
    """
    filelist = glob.glob(ls_command)
    for f in filelist:
        if not os.path.isfile(f):
            raise Exception("Datafile %s not found!" % f)
    return str(filelist).replace('\'', '\"').replace(" ", "")


def inference_simple(cfg, blobs, net, num_test=10, scope=None, test_image=None, **net_args):
    """
    Assumes blobs[i] is a list of blobs (crops).
    Returns inference[i] = list of results for each crop.
    """
    net.init_placeholders(**net_args)
    if scope is None:
        net.create_architecture(is_training=False)
    else:
        net.create_architecture(is_training=False, scope=scope)
    if test_image is None:
        test_image = net.test_image

    inference = []
    duration = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        load_weights(cfg, sess)
        for i in range(num_test):
            inference_blob = []
            for j, blob in enumerate(blobs[i]):
                start = time.time()
                summary, results = test_image(sess, blob)
                end = time.time()
                duration.append(end - start)
                inference_blob.append(results)
            inference.append(inference_blob)
    print("Average duration of inference = %f s" % np.array(duration).mean())
    return inference


def inference(cfg):
    """
    Inference for `ppn`, `base`, `full`.
    Retrieves in a loop all the data first and stores it.
    Memory issues could arise if too many steps are requested.
    """
    # if cfg.WEIGHTS_FILE_BASE is None or cfg.WEIGHTS_FILE_PPN is None:
    #     raise Exception("Need both weights files for full inference.")

    if not os.path.isdir(cfg.DISPLAY_DIR):
        os.makedirs(cfg.DISPLAY_DIR)

    num_test = cfg.MAX_STEPS
    inference_base, inference_ppn, blobs = [], [], []
    weights_file_ppn = cfg.WEIGHTS_FILE_PPN
    crop_algorithm = cropping_algorithms[cfg.CROP_ALGO](cfg)

    # 0. Loop to retrieve all the data first
    # --------------------------------------
    # Memory issues could arise here if we ask for too many steps.
    print("Retrieving data...")
    train_data, data = get_data(cfg)
    patch_centers_list, patch_sizes_list = [], []
    for i in range(num_test):
        blob = data.forward()
        # Cropping pre-processing
        patch_centers, patch_sizes = None, None
        if cfg.ENABLE_CROP:
            batch_blobs, patch_centers, patch_sizes = crop_algorithm.process(blob)
            patch_centers_list.append(patch_centers)
            patch_sizes_list.append(patch_sizes)
        else:
            batch_blobs = [blob]
        blobs.append(batch_blobs)
    print("Done.")

    if cfg.PROFILE:
        print('WARNING PROFILING ENABLED')
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # print(getargspec(self.sess.run))
        run_metadata = tf.RunMetadata()
        old_run = tf.Session.run
        new_run = lambda self, fetches, feed_dict=None: old_run(self, fetches, feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        tf.Session.run = new_run

    # 1. Run inference of all the networks.
    # -------------------------------------
    # Depending on cfg.NET value, build and run the networks inferences.

    # First base
    inference_base = None
    if cfg.NET in ['full', 'base']:
        print("Base network...")
        cfg.WEIGHTS_FILE_PPN = None
        net_base = basenets[cfg.BASE_NET](cfg=cfg)
        inference_base = inference_simple(cfg, blobs, net_base, num_test=num_test)
        print("Done.")

    tf.reset_default_graph()

    # Then PPN
    inference_ppn = None
    if cfg.NET in ['full', 'ppn', 'ppn_ext']:
        print("PPN network...")
        cfg.WEIGHTS_FILE_PPN = weights_file_ppn
        net_ppn = PPN(cfg=cfg, base_net=basenets[cfg.BASE_NET])
        inference_ppn = inference_simple(cfg, blobs, net_ppn, num_test=num_test)
        print("Done.")

    # Small UResNet (try to get better precision after PPN?)
    inference_small_uresnet = None
    if cfg.NET == 'ppn_ext':
        print("Small UResNet + PPN network...")
        # FIXME better way to control the number of crops here?
        crops = crop_proposals(cfg, net_ppn.image_placeholder, net_ppn._predictions['im_proposals'])[:512]
        print(crops)
        # Cannot use tf.train.batch because the call to tf.train.start_queue_runners
        # requires image placeholder to be fed already
        #crops = tf.train.batch([crops], 1, shapes=[tf.TensorShape((cfg.CROP_SIZE, cfg.CROP_SIZE))], dynamic_pad=True, allow_smaller_final_batch=False, enqueue_many=True)
        net_uresnet = UResNet(cfg=cfg, N=cfg.CROP_SIZE)
        # FIXME remove dependency on labels at test time
        net_args = {
            'image': tf.reshape(crops, (-1, cfg.CROP_SIZE, cfg.CROP_SIZE, 1)),
            'labels': tf.cast(tf.reshape(crops, (-1, cfg.CROP_SIZE, cfg.CROP_SIZE)),
                           dtype=tf.int32)
        }

        def test_image_small_uresnet(sess, blob):
            results = sess.run([
                crops,
                net_uresnet._predictions,
                net_uresnet._scores
            ], feed_dict={net_ppn.image_placeholder: blob['data'], net_ppn.gt_pixels_placeholder: blob['gt_pixels']})
            return None, {'crops': results[0], 'predictions_small': results[1], 'scores_small': results[2]}

        inference_small_uresnet = inference_simple(cfg, blobs, net_uresnet,
                                                   num_test=num_test,
                                                   scope='small_uresnet',
                                                   test_image=test_image_small_uresnet,
                                                   **net_args)
        print("Done.")

    # 2. Display all inference results
    # --------------------------------
    # Also computes associated metrics if relevant.

    print("Saving displays...")
    if inference_ppn is not None:
        metrics_ppn = PPNMetrics(cfg, dim1=net_ppn.dim1, dim2=net_ppn.dim2)
    if inference_base is not None and cfg.BASE_NET == 'uresnet':
        metrics_uresnet = UResNetMetrics(cfg)

    real_step = 0
    final_results = []
    for i in range(num_test):
        if cfg.ENABLE_CROP:
            N = cfg.IMAGE_SIZE
            cfg.IMAGE_SIZE = cfg.SLICE_SIZE
        final_blob_results = []
        for j, blob in enumerate(blobs[i]):
            print("%d - %d/%d" % (i, j, len(blobs[i])))
            real_step += 1
            results = {}
            if inference_base is not None:
                results.update(inference_base[i][j])
            if inference_ppn is not None:
                results.update(inference_ppn[i][j])
            final_blob_results.append(results)

            if cfg.NET == 'full':
                display_ppn_uresnet(
                    blob,
                    cfg,
                    index=i,
                    directory=os.path.join(cfg.DISPLAY_DIR, 'demo_full'),
                    **results
                )
                metrics_ppn.add(blob, results)
                metrics_uresnet.add(blob, results)
            elif cfg.NET in ['ppn', 'ppn_ext']:
                display(
                    blob,
                    cfg,
                    index=real_step,
                    dim1=net_ppn.dim1,
                    dim2=net_ppn.dim2,
                    directory=os.path.join(cfg.DISPLAY_DIR, 'demo'),
                    **results
                )
                metrics_ppn.add(blob, results)
            elif cfg.NET == 'base' and cfg.BASE_NET == 'uresnet':
                display_uresnet(blob, cfg,
                                index=real_step,
                                directory=os.path.join(cfg.DISPLAY_DIR, 'demo'),
                                **results)
                metrics_uresnet.add(blob, results)
            else:  # No display function available, just print results.
                print(blob, results)
            if cfg.NET == 'ppn_ext':
                N = cfg.IMAGE_SIZE
                cfg.IMAGE_SIZE = cfg.CROP_SIZE
                for k, crop in enumerate(results['crops']):
                    blob_j = {'data': np.reshape(crop, (1, cfg.CROP_SIZE, cfg.CROP_SIZE, 1))}
                    # FIXME generate labels from gt ?
                    blob_j['labels'] = blob_j['data'][:, :, :, 0]
                    pred = np.reshape(results['predictions_small'][k], (1, cfg.CROP_SIZE, cfg.CROP_SIZE))
                    scores = np.reshape(results['scores_small'][k], (1, cfg.CROP_SIZE, cfg.CROP_SIZE))
                    display_uresnet(blob_j, cfg,
                                    index=real_step*100+k,
                                    name='display_small',
                                    directory=os.path.join(cfg.DISPLAY_DIR, 'demo_small'),
                                    vmin=0,
                                    vmax=1,
                                    predictions=pred,
                                    scores=scores)

                cfg.IMAGE_SIZE = N

            # 3. Ad-hoc clustering
            # --------------------
            # FIXME why is this reshape necessary?
            results['predictions'] = results['predictions'][np.newaxis, ...]
            if cfg.NET != 'base':
                cluster(cfg, blob, results, i, name='cluster_full', directory=os.path.join(cfg.DISPLAY_DIR, 'cluster_full'))

        if cfg.ENABLE_CROP:
            cfg.IMAGE_SIZE = N
            final_blob_results = crop_algorithm.reconcile(final_blob_results,
                                                     patch_centers_list[i],
                                                     patch_sizes_list[i])

            # display(blob,
            #          cfg,
            #          index=i,
            #          name='display_train_final',
            #          directory=os.path.join(self.cfg.DISPLAY_DIR,
            #                                 'train'),
            #          **final_results)
        else:
            final_blob_results = final_blob_results[0]
        final_results.append(final_blob_results)


    print('Plot metrics...')
    if (cfg.NET == 'base' and cfg.BASE_NET == 'uresnet') or cfg.NET == 'full':
        metrics_uresnet.plot()
    elif cfg.NET in ['ppn', 'full']:
        metrics_ppn.plot()
    print("Done.")

    if cfg.PROFILE:
        # Create the Timeline object, and write it to a json
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(cfg.PROFILE_NAME, 'w') as f:
            f.write(ctf)
            print("Wrote timeline to %s" % cfg.PROFILE_NAME)

        # # Print to stdout an analysis of the memory usage and the timing information
        # # broken down by python codes.
        # ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
        # opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
        #     ).with_node_names(show_name_regexes=['*']).build()
        #
        # tf.profiler.profile(
        #     tf.get_default_graph(),
        #     run_meta=run_metadata,
        #     cmd='code',
        #     options=opts)
        #
        # # Print to stdout an analysis of the memory usage and the timing information
        # # broken down by operation types.
        # tf.profiler.profile(
        #     tf.get_default_graph(),
        #     run_meta=run_metadata,
        #     cmd='op',
        #     options=tf.profiler.ProfileOptionBuilder.time_and_memory())
    del train_data
    del data
    return blobs, final_results


def crop_step(crops, coords0, coords1, data, N):
    j = tf.shape(crops)[0]-1
    j = tf.Print(j, [j])
    padding = []
    dim = coords0.get_shape()[1]
    for d in range(dim):
        pad = tf.maximum(N - (coords1[j, d] - coords0[j, d]), 0)
        if coords0[j, d] == 0.0:
            padding.append((pad, 0))
        else:
            padding.append((0, pad))
    new_crop = tf.pad(data[0, coords0[j, 0]:coords1[j, 0], coords0[j, 1]:coords1[j, 1], 0], padding, mode='constant')
    crops = tf.concat([crops, [new_crop]], axis=0)
    return crops, coords0, coords1, data, N


def crop_proposals(cfg, data, proposals):
    N = cfg.CROP_SIZE
    coords0 = tf.cast(tf.floor(proposals - N/2.0), tf.int32)
    coords1 = tf.cast(tf.floor(proposals + N/2.0), tf.int32)
    dim = 3 if cfg.DATA_3D else 2
    smear = tf.random_uniform((dim,), minval=-3, maxval=3, dtype=tf.int32)
    coords0 = tf.clip_by_value(coords0 + smear, 0, cfg.IMAGE_SIZE-1)
    coords1 = tf.clip_by_value(coords1 + smear, 0, cfg.IMAGE_SIZE-1)
    crops = tf.zeros((1, N, N))#tf.zeros((tf.shape(coords0)[0], N, N))
    results = tf.while_loop(lambda crops, coords0, *args: tf.shape(crops)[0] <= tf.shape(coords0)[0],
                            crop_step, [crops, coords0, coords1, data, N],
                            shape_invariants=[tf.TensorShape((None, N, N)), coords0.get_shape(), coords1.get_shape(), data.get_shape(), tf.TensorShape(())])
    crops = results[0][1:, :, :]
    return crops


def cluster(cfg, blob, results, index, name='cluster', directory=None):
    """
    Ad-hoc clustering algorithm. Can use UResNet predictions as a mask for
    to cluster track and shower separately, if results includes `predictions`
    key. Erases a 7x7 window around each point predicted by PPN in the data,
    then applies DBSCAN algorithm to perform rough clustering of track/shower
    instances.
    """
    data = blob['data']
    WINDOW_SIZE = 7
    # Hide window around each proposal
    for p in results['im_proposals']:
        p1 = (p - WINDOW_SIZE/2).astype(int)
        p2 = (p + WINDOW_SIZE/2).astype(int)
        if cfg.DATA_3D:
            data[0, p1[0]:p2[0], p1[1]:p2[1], p1[2]:p2[2], 0] = 0.0
        else:
            data[0, p1[0]:p2[0], p1[1]:p2[1], 0] = 0.0

    if 'predictions' in results:  # UResNet mask
        predictions = results['predictions'][0, ...]
        predictions[data[0, ..., 0] == 0.0] = 0.0  # mask with data
        track_voxels = np.argwhere(predictions == 1)  # track
        shower_voxels = np.argwhere(predictions == 2)  # track
        db_track, db_shower = [], []
        if track_voxels.shape[0]:
            db_track = DBSCAN(eps=2.5 if cfg.DATA_3D else 2.0,
                              min_samples=10).fit_predict(track_voxels)
        if shower_voxels.shape[0]:
            db_shower = DBSCAN(eps=2.5 if cfg.DATA_3D else 2.0,
                               min_samples=10).fit_predict(shower_voxels)
            db_shower = db_shower + len(np.unique(db_track))  # offset labels
        voxels = np.concatenate([track_voxels, shower_voxels], axis=0)
        voxels = np.flip(voxels, axis=1)
        db = np.concatenate([db_track, db_shower], axis=0)
    else:
        voxels, _ = extract_voxels(data[0, ..., 0])
        voxels = np.flip(voxels, axis=1)
        db = DBSCAN(eps=2.5 if cfg.DATA_3D else 2.0,
                    min_samples=10).fit_predict(voxels)

    print("Clusters: ", np.unique(db))
    new_blob = {}
    new_blob['data'] = data
    new_blob['voxels'] = voxels
    new_blob['voxels_value'] = db
    # display_uresnet(new_blob, cfg,
    #                 compute_voxels=False,
    #                 directory=directory,
    #                 vmin=np.amin(db),
    #                 vmax=np.amax(db),
    #                 index=index)
    kwargs = {}
    if cfg.DATA_3D:
        kwargs['projection'] = '3d'

    display_blob(new_blob, cfg, directory=directory, index=index, cmap='tab10', **kwargs)

# if __name__ == '__main__':
#     inference(cfg)
