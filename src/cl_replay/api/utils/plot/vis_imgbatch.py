import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_path", default="xs_batch.npy", type=str,
                        help="rel. path of sub dir with a mini-batch in .npy format")
    parser.add_argument("--out", default = "./results/input_out", type=str, help="output file name")
    parser.add_argument("--num_samples", default = 1, type=int, help="how many samples to visualize")
    parser.add_argument("--num_cols", default = 1, type=int, help="how many cols")
    parser.add_argument("--num_rows", default = 1, type=int, help="how many rows")

    FLAGS = parser.parse_args()

    xs = np.load(FLAGS.batch_path)
    images      = xs[:FLAGS.num_samples]
    #labels     = ys[:FLAGS.num_samples]

    print(FLAGS.num_samples, FLAGS.num_rows, FLAGS.num_cols)

    fig, axes = plt.subplots(FLAGS.num_rows, FLAGS.num_cols, figsize=(FLAGS.num_cols, FLAGS.num_rows))
    for i in range(FLAGS.num_samples):
        if FLAGS.num_rows != 1:
            ax = axes[i//FLAGS.num_cols, i%FLAGS.num_cols]
        else:
            ax = axes[i%FLAGS.num_cols]
        ax.imshow(images[i], cmap='gray') # cmap='gray_r' for color swap

        #ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.set_aspect('equal')
        #ax.set_title('Label: {}'.format(labels[i]))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.axis('off')
    #plt.tight_layout(h_pad=0.2, w_pad=0.8)
    plt.savefig(FLAGS.out, transparent=True, bbox_inches='tight')