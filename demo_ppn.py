# *-* encoding: utf-8 *-*
# Demo for PPN
# Usage: python demo_ppn.py model.ckpt

def display(blob, ppn1_proposals, ppn1_labels, rois, ppn2_proposals, ppn2_positives, index=0):
    #fig, ax = plt.subplots(1, 1, figsize=(18,18), facecolor='w')
    #ax.imshow(blob['data'][0,:,:,0], interpolation='none', cmap='hot', origin='lower')
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.imshow(blob['data'][0,:,:,0], cmap='coolwarm', interpolation='none', origin='lower')
    for i in range(len(ppn1_labels)):
        if ppn1_labels[i] == 1:
            plt.plot([ppn1_proposals[i][1]*32.0], [ppn1_proposals[i][0]*32.0], 'y+')
            coord = np.floor(ppn1_proposals[i])*32.0
            print(floor(coord[1]), floor(coord[0]))
            ax.add_patch(
                patches.Rectangle(
                    (coord[1], coord[0]),
                    32, # width
                    32, # height
                    #fill=False,
                    #hatch='\\',
                    facecolor='green',
                    alpha = 0.5,
                    linewidth=2.0,
                    edgecolor='red',
                )
            )

    for roi in rois:
        print(roi[1]*32.0, roi[0]*32.0)
        ax.add_patch(
            patches.Rectangle(
                (roi[1]*32.0, roi[0]*32.0),
                32, # width
                32, # height
                #fill=False,
                #hatch='\\',
                facecolor='green',
                alpha = 0.3,
                linewidth=2.0,
                edgecolor='black',
            )
        )
    for i in range(len(rois)):
        for j in range(16):
            if ppn2_positives[i*16+j]:
                plt.plot([ppn2_proposals[i][1]*8.0+rois[i][1]*32.0], [ppn2_proposals[i][0]*8.0+rois[i][0]*32.0], 'r+')
                coord = np.floor(ppn2_proposals[i])*8.0 + rois[i]*32.0
                print(floor(coord[1]), floor(coord[0]))
                ax.add_patch(
                    patches.Rectangle(
                        (coord[1], coord[0]),
                        8, # width
                        8, # height
                        #fill=False,
                        #hatch='\\',
                        facecolor='yellow',
                        alpha = 0.8,
                        linewidth=2.0,
                        edgecolor='orange',
                    )
                )
    #plt.imsave('display.png', blob['data'][0,:,:,0])
    plt.savefig('display%d.png' % index)

def inference():
    toydata = ToydataGenerator(N=512, max_tracks=5, max_kinks=2, max_track_length=200)

    net = PPN()
    net.create_architecture()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, sys.argv[1])
        for i in range(10):
            blob = toydata.forward()
            im_proposals, ppn1_proposals, labels_ppn1, rois, ppn2_proposals, ppn2_positives = net.test_image(blob)
            display(blob, ppn1_proposals, labels_ppn1, rois, ppn2_proposals, ppn2_positives, index=i)

if __name__ == '__main__':
    inference()
