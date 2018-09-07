from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from faster_particles.metrics.metrics import Metrics


class UResNetMetrics(Metrics):
    def __init__(self, cfg):
        super(UResNetMetrics, self).__init__(cfg)
        self.acc_all, self.acc_nonzero = [], []
        self.label_softmax_mean = []
        self.label_softmax_std = []
        self.num_voxels = []
        self.label_softmax_nonzero_mean = []
        self.label_softmax_nonzero_std = []
        self.class_npx, self.class_acc = {}, {}
        self.class_score_mean, self.class_score_std = {}, {}
        for class_label in np.arange(self.cfg.NUM_CLASSES):
            self.class_npx[class_label] = []
            self.class_acc[class_label] = []
            self.class_score_mean[class_label] = []
            self.class_score_std[class_label] = []

    def add(self, blob, results):
        blob['labels'] = np.squeeze(blob['labels'])
        results['predictions'] = np.squeeze(results['predictions'])
        results['softmax'] = np.squeeze(results['softmax'])

        acc_all = np.mean(results['predictions'] == blob['labels'])

        # Nonzero accuracy
        nonzero_px = np.where(blob['labels'] > 0)
        nonzero_prediction = results['predictions'][nonzero_px]
        nonzero_label = blob['labels'][nonzero_px]
        acc_nonzero = np.mean(nonzero_prediction == nonzero_label)

        # Softmax average
        label_softmax = np.zeros_like(blob['labels'])
        for class_label in np.arange(self.cfg.NUM_CLASSES):
            class_mask = np.where(blob['labels'] == class_label)
            class_softmax = results['softmax'][..., class_label]
            label_softmax[class_mask] = class_softmax[class_mask]

        self.acc_all.append(acc_all)
        self.label_softmax_mean.append(label_softmax.mean())
        self.label_softmax_std.append(label_softmax.std())
        self.num_voxels.append((blob['labels'] > 0).astype(np.int32).sum())
        self.acc_nonzero.append(acc_nonzero)
        self.label_softmax_nonzero_mean.append(label_softmax[nonzero_px].mean())
        self.label_softmax_nonzero_std.append(label_softmax[nonzero_px].std())

        # Class-wise accuracy, mean/std score value
        for class_label in np.arange(self.cfg.NUM_CLASSES):
            # Create a mask to select this class
            class_mask = np.where(blob['labels'] == class_label)
            # compute pixel count for this class
            npx = blob['labels'][class_mask].size
            # If npx > 0 compute class accuracy and softmax score mean/std
            class_acc, class_score_mean, class_score_std = -1., -1., -1.
            if npx:
                # compute class accuracy
                class_prediction = results['predictions'][class_mask]
                class_acc = np.mean(class_prediction == class_label)
                # compute softmax score mean value
                class_score = results['softmax'][..., class_label][class_mask]
                class_score_mean = class_score.mean()
                class_score_std = class_score.std()
            self.class_npx[class_label].append(npx)
            self.class_acc[class_label].append(class_acc)
            self.class_score_mean[class_label].append(class_score_mean)
            self.class_score_std[class_label].append(class_score_std)

    def plot(self):
        # Save data
        np.savetxt(os.path.join(self.dir, "acc_all.csv"), self.acc_all, delimiter=",")
        np.savetxt(os.path.join(self.dir, "acc_nonzero.csv"), self.acc_nozero, delimiter=",")
        np.savetxt(os.path.join(self.dir, "label_softmax_mean.csv"), self.label_softmax_mean, delimiter=",")
        np.savetxt(os.path.join(self.dir, "label_softmax_std.csv"), self.label_softmax_std, delimiter=",")
        np.savetxt(os.path.join(self.dir, "num_voxels.csv"), self.num_voxels, delimiter=",")
        np.savetxt(os.path.join(self.dir, "label_softmax_nonzero_mean.csv"), self.label_softmax_nonzero_mean, delimiter=",")
        np.savetxt(os.path.join(self.dir, "label_softmax_nonzero_std.csv"), self.label_softmax_nonzero_std, delimiter=",")
        for class_label in np.arange(self.cfg.NUM_CLASSES):
            np.savetxt(os.path.join(self.dir, "class_npx_%d.csv" % class_label), self.class_npx[class_label], delimiter=",")
            np.savetxt(os.path.join(self.dir, "class_acc_%d.csv" % class_label), self.class_acc[class_label], delimiter=",")
            np.savetxt(os.path.join(self.dir, "class_score_mean_%d.csv" % class_label), self.class_score_mean[class_label], delimiter=",")
            np.savetxt(os.path.join(self.dir, "class_score_std_%d.csv" % class_label), self.class_score_std[class_label], delimiter=",")
        self.plot_acc_all()
        self.plot_acc_nonzero()
        self.plot_label_softmax_mean()
        self.plot_label_softmax_std()
        self.plot_num_voxels()
        self.plot_label_softmax_nonzero_mean()
        self.plot_label_softmax_nonzero_std()
        self.plot_class_npx()
        self.plot_class_acc()
        self.plot_class_score_mean()
        self.plot_class_score_std()

    def plot_acc_all(self):
        self.make_plot(
            self.acc_all,
            bins=np.linspace(0, 1, 20),
            xlabel="Accuracy",
            ylabel="#Images",
            filename='acc_all.png'
        )

    def plot_acc_nonzero(self):
        self.make_plot(
            self.acc_nonzero,
            bins=np.linspace(0, 1, 20),
            xlabel="Accuracy on nonzero pixels",
            ylabel="#Images",
            filename='acc_nonzero.png'
        )

    def plot_label_softmax_mean(self):
        self.make_plot(
            self.label_softmax_mean,
            bins=20,
            xlabel="Label softmax mean",
            ylabel="#Images",
            filename='label_softmax_mean.png'
        )

    def plot_label_softmax_std(self):
        self.make_plot(
            self.label_softmax_std,
            bins=20,
            xlabel="Label softmax std",
            ylabel="#Images",
            filename='label_softmax_std.png'
        )

    def plot_num_voxels(self):
        self.make_plot(
            self.num_voxels,
            bins=20,
            xlabel="Number of nonzero voxels",
            ylabel="#Images",
            filename='num_voxels.png'
        )

    def plot_label_softmax_nonzero_mean(self):
        self.make_plot(
            self.label_softmax_nonzero_mean,
            bins=20,
            xlabel="Label softmax mean on nonzero voxels",
            ylabel="#Images",
            filename='label_softmax_nonzero_mean.png'
        )

    def plot_label_softmax_nonzero_std(self):
        self.make_plot(
            self.label_softmax_nonzero_std,
            bins=20,
            xlabel="Label softmax std on nonzero voxels",
            ylabel="#Images",
            filename='label_softmax_nonzero_std.png'
        )

    def plot_class_npx(self):
        for class_label in np.arange(self.cfg.NUM_CLASSES):
            self.make_plot(
                self.class_npx[class_label],
                bins=20,
                xlabel="Number of nonzero voxels for class %d" % class_label,
                ylabel="#Images",
                filename='class_npx.png'
            )

    def plot_class_acc(self):
        for class_label in np.arange(self.cfg.NUM_CLASSES):
            self.make_plot(
                self.class_acc[class_label],
                bins=20,
                xlabel="Accuracy for class %d" % class_label,
                ylabel="#Images",
                filename='class_acc.png'
            )

    def plot_class_score_mean(self):
        for class_label in np.arange(self.cfg.NUM_CLASSES):
            self.make_plot(
                self.class_score_mean[class_label],
                bins=20,
                xlabel="Mean score for class %d" % class_label,
                ylabel="#Images",
                filename='class_score_mean.png'
            )

    def plot_class_score_std(self):
        for class_label in np.arange(self.cfg.NUM_CLASSES):
            self.make_plot(
                self.class_score_std[class_label],
                bins=20,
                xlabel="Std of score for class %d" % class_label,
                ylabel="#Images",
                filename='class_score_std.png'
            )
