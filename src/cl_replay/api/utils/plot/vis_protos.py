import os
import sys, gzip, pickle
from pathlib import Path

import numpy              as np
import matplotlib         as mp
import matplotlib.pyplot  as plt, sys, math
import imageio.v2 as imageio

from matplotlib   import cm
from argparse   import ArgumentParser

mp.use("Agg") # so as to not try to invoke X on systems without it


def getIndexArray(w_in, h_in, c_in):
    indicesOneSample  = np.zeros([w_in*h_in*c_in],dtype=np.int32)
    tileX             = int(math.sqrt(c_in)) ;  tileY = tileX
    tilesX            = w_in ;                  tilesY = c_in
    sideX             = tileX * w_in ;          sideY = tileY * h_in

    for inIndex in range(0,w_in * h_in * c_in):
        tileIndexFlat = inIndex // c_in
        tileIndexX    = tileIndexFlat % tilesX
        tileIndexY    = tileIndexFlat // tilesX

        tilePosFlat   = inIndex % (tileX*tileY)
        tilePosY      = tilePosFlat // tileX
        tilePosX      = tilePosFlat % tileX

        posY          = tileIndexY * tileY + tilePosY
        posX          = tileIndexX * tileX + tilePosX
        outIndex      = sideX  * posY + posX

        indicesOneSample [outIndex] = inIndex

    return indicesOneSample


def vis_img_data(prefix, prefix_path=None, suffix=0, **kwargs):
    '''
    visualizes data saved by GMM Layers --> GMM_Layer.py
    Can vis, per component, depending on parameter "--what":
    - centroids (mus)
    - diagonal sigmas (sigmas)
    - loading matrix rows (loadingMatrix)
    - convGMM centroids arranged in visually intuitive way (organizedWeights)
    Over all visualizations the pi value of that comp. can be overlaid.
    '''
    pi_file         = kwargs.get('pi_file',         "pis.npy")
    mu_file         = kwargs.get('mu_file',         "mus.npy")
    sigma_file      = kwargs.get('sigma_file',      "sigmas.npy")
    dataset_file    = kwargs.get('dataset_file',    "")
    channels        = kwargs.get('channels',        1)
    what            = kwargs.get('what',            "mus")
    vis_pis         = kwargs.get('vis_pis',         False)
    vis_cons        = kwargs.get('vis_cons',        False)
    vis_cons_mode   = kwargs.get('vis_cons_mode',   "eval")
    img_sequence    = kwargs.get('img_sequence',    False)
    out             = kwargs.get('out',             None)
    x               = kwargs.get('x',               0)
    y               = kwargs.get('y',               0)
    l               = kwargs.get('y',               0)
    proto_size      = kwargs.get('proto_size',      [-1, -1])
    prev_layer_hwc  = kwargs.get('prev_layer_hwc',  [-1, -1, -1])
    cur_layer_hwc   = kwargs.get('cur_layer_hwc',   [-1, -1, -1])
    filter_size     = kwargs.get('filter_size',     [-1, -1])
    pad             = kwargs.get('pad',             0)
    w_pad           = kwargs.get('w_pad',           0)
    h_pad           = kwargs.get('h_pad',           0)
    disp_range      = kwargs.get('disp_range',      [-100., +100.])
    clip_range      = kwargs.get('clip_range',      [-100., +100.])

    print(f'IN: {prefix_path}')
    if what == 'loadingMatrix':
        sigma_file = 'gammas.npy'

    if vis_cons:
        train_con_file   = 'train_connectivities.npy'
        test_con_file    = 'eval_connectivities.npy'
        try:
            train_cons      = np.load(prefix_path + train_con_file)
            vis_train_cons  = True
            component_count = train_cons.shape[1]
        except OSError: vis_train_cons = False
        try:
            eval_cons       = np.load(prefix_path + test_con_file)
            vis_eval_cons   = True
            component_count = eval_cons.shape[1]
        except OSError: vis_eval_cons = False

    print(f'FILE(S) LOCATION: {prefix_path}')
    print(f'FILE(S) PREFIX: {prefix}')

    if out:
        if img_sequence:
            seq_path        = sequence_path
            task_id         = seq_path[seq_path.find('_')+1:]
            save_path       = os.path.join(out, f'{prefix}{task_id}_E{suffix}_mus.png')
            print(f'TASK ID: {task_id}')
        else:
            save_path       = os.path.join(out, 'mus.png')
        print(f'SAVE: {save_path}')
    
    protos = np.load(prefix_path + mu_file)
    print("RAW PROTOs SHAPE: ", protos.shape)
    pis = sigmas = None
    if len(protos.shape) == 4: # sampling file, single npy with dims N,d*d
        _n,_h,_w,_c = protos.shape
        _d          = _h*_w*_c ;
        pis         = np.zeros([_n,_d])
        sigmas      = np.zeros([_n,_d])
        protos      = protos.reshape(_n,_d)
    else:
        pis     = np.load(prefix_path + pi_file)[0, y, x]
        sigmas  = np.load(prefix_path + sigma_file)[0, y, x]
        protos  = protos[0, y, x]
    print("RESHAPED PROTOs SHAPE: ", protos.shape)

    n     = int(math.sqrt(protos.shape[0]))
    imgW  = int(math.sqrt(protos.shape[1] / channels))
    imgH  = int(math.sqrt(protos.shape[1] / channels))
    d_    = int(math.sqrt(protos.shape[1] / channels))

    if proto_size[0] != -1: imgH, imgW = proto_size
    print("imgH, imgW: ", imgH, imgW)
    print("ND: ", n, d_)

    h_in, w_in, c_in    = prev_layer_hwc
    h_out, w_out, c_out = cur_layer_hwc
    pH, pW              = filter_size

    print("IN: ", h_in, w_in, c_in)
    print("OUT: ", h_out, w_out, c_out)
    print("FILTER SIZE: ", pH, pW)

    indices = None
    if h_in != -1:
        indices = getIndexArray(h_in, w_in, c_in, h_out, w_out, c_out, pH, pW)
        print("INDICES: ", indices.shape, indices.min(), indices.max())

    if what == "mus2D":
        f,ax = plt.subplots(1,1)
        data = None
        if dataset_file != "":
            with gzip.open(dataset_file) as f:
                data = pickle.load(f)["data_test"]
            print("LOADED DATA SHAPE (mus2D): ", data.shape)
        if data is not None: ax.scatter(data[:,0], data[:,1])
        ax.scatter(protos[:,0],protos[:,1])
        plt.tight_layout(pad=0.1, h_pad=.0, w_pad=-10.)
        plt.savefig(out)
        sys.exit(0)

    fig = plt.figure(frameon=False)

    '''
    f, axes = plt.subplots(1, 1)
    disp = protos[15]

    refmin = disp.min()
    refmax = disp.max()
    
    img = plt.imshow(disp.reshape(imgH, imgW, channels), vmin=refmin, vmax=refmax)
    '''    
    
    f = axes = None
    #f, axes = plt.subplots(n, n, gridspec_kw={'wspace':0, 'hspace':0})
    f, axes = plt.subplots(n, n)

    if n == 1:
        f = np.array([f])
        axes = np.array([axes])
    
    axes = axes.ravel()
    index = -1

    exp_pi = np.exp(pis)
    sm = exp_pi/exp_pi.sum()

    for (dir_, ax_, pi_, sig_) in zip(protos, axes, sm, sigmas):
        index += 1

        disp = dir_
        if what == "precs_diag":      disp = sig_
        if what == "loadingMatrix":   disp = sig_[:,l]

        if what == "organizedWeights" and indices is not None:
            dispExp = (disp.reshape(h_in, w_in, c_in))
            disp = dispExp
            disp = disp.ravel()[indices]

        #disp    = np.clip(disp, clip_range[0], clip_range[1])
        refmin  = disp.min() if disp_range[0] == -100 else disp_range[0]
        refmax  = disp.max() if disp_range[1] == +100 else disp_range[1]

        # INFO: This is interesting to see unconverged components
        # print(f'index: {index}; min/max: {disp.min()}/{disp.max()};' + 
        #     f'ref_min/max: {refmin}/{refmax}, disp shape: {disp.shape}, {channels} {imgH} {imgW}')

        if vis_cons:
            def vis_connections(comp_con):
                component_con   = comp_con[:, index]
                #print(f'component activity (row), 0-10 classes for one component:\n{component_con}\n')
                top_K_indices   = np.argpartition(component_con, -1)[-3:]
                top_K_vals      = np.array(component_con[top_K_indices], dtype=np.float32)
                top_K_vals      /= top_K_vals.sum(dtype=np.float32)
                top_K_vals      = top_K_vals.round(2)
                #print('component TOP-K: ', top_K_indices, top_K_vals)
                colors = ['yellow', 'orange', 'red']
                if imgH > 20:
                    first = 6
                    increment = 8
                    x = 0

                if imgH < 20:
                    first = 2
                    increment = 1
                    x = 0

                if imgH < 14:
                    first = 1
                    increment = 1.5
                    x = 0

                if component_count <= 36:
                    for K in range(2, -1, -1):
                        ax_.text(x, first, 'M{:} {:.2f}'.format(top_K_indices[K], top_K_vals[K]), fontsize=8, c=colors[K], alpha=0.5,
                                    horizontalalignment='center', verticalalignment='center',
                                    fontfamily='sans-serif', fontstretch='ultra-condensed', fontweight='bold')
                        #ax_.text.set(alpha=0.5, horizontalalignment='center', fontvariant='small-caps')
                        first += increment
                    w_pad   = .5
                    h_pad   = .5
                    pad     = .1
                else:
                    for K in range(2, -1, -1):
                        ax_.text(x, first, '[{:}] {:.2f}'.format(top_K_indices[K], top_K_vals[K]), fontsize=6, c=colors[K], alpha=0.5,
                                    horizontalalignment='center', verticalalignment='center',
                                    fontfamily='sans-serif', fontstretch='ultra-condensed', fontweight='bold')
                        first += increment
                    w_pad   = .2
                    h_pad   = .2
                    pad     = .05

            if vis_cons_mode == 'train' and vis_train_cons:   vis_connections(train_cons)
            if vis_cons_mode == 'eval' and vis_eval_cons:     vis_connections(eval_cons)

        ax_.imshow(disp.reshape(imgH, imgW, channels) if channels == 3 else disp.reshape(imgH,imgW), vmin=refmin, vmax=refmax, cmap=cm.bone)

        if vis_pis == True:
            ax_.text(-5, 1, "%.03f" % (pi_), fontsize=5, c="black", bbox=dict(boxstyle="round", fc=(1, 1, 1), ec=(.5, .5, .5)))

        #ax_.set_aspect('auto')
        ax_.tick_params( # disable labels and ticks
                axis        = 'both',
                which       = 'both',
                bottom      = False ,
                top         = False ,
                left        = False ,
                right       = False ,
                labelbottom = False ,
                labelleft   = False ,
        )
    #plt.subplots_adjust(left=0,right=0.1,bottom=0,top=0.1,wspace=0.,hspace=0.)
    plt.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)
    if out: plt.savefig(save_path, transparent=True)


if __name__ == "__main__":
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')
    ''' Example usage:
    python3 $HOME/sccl/src/cl_replay/api/utils/vis.py \
    --sequence_path "/home/ak/exp-results/qgmm_lf/protos/protos_T1" \
    --prefix "qgmm_lf_L2_GMM_" \
    --out "$HOME/Desktop/eval-exp/protos_out/qgmm_lf/protos_T1" \
    --epoch -1 \
    --vis_cons False \
    --vis_cons_mode "eval" \
    --channels 3 \
    --proto_size 9 100          
    '''

    parser = ArgumentParser()

    parser.add_argument("--channels",       default = 1,                        type=int,   help = "If u r visualizing centroids that come from color images (SVHN, Fruits), please specify 3 here!")
    parser.add_argument("--y",              default = 0,                        type=int,   help = "PatchY index for convGMMs")
    parser.add_argument("--x",              default = 0,                        type=int,   help = "PatchX index for convGMMs")
    parser.add_argument("--l",              default = 0,                        type=int,   help = "row of MFA loading matrix")
    parser.add_argument("--what",           default = "mus",                    type=str,   choices=["mus2D", "mus", "precs_diag", "organizedWeights","loadingMatrix"],  help="Visualize centroids or precisions°")
    parser.add_argument("--prefix",         default = "gmm_layer_",             type=str,   help="Prefix for file names")
    parser.add_argument("--vis_pis",        default = False,                    type=eval,  help="True or False depending on whether you want the weights drawn on each component")
    parser.add_argument("--vis_cons",       default = False,                    type=eval,  help="True or False depending on whether you want the top K component/classification connectivities drawn on each component")
    parser.add_argument("--vis_cons_mode",  default = "eval",                   type=str,   help="visualize test/train component mapping")
    parser.add_argument("--cur_layer_hwc",  default = [-1, -1, -1],             type=int,   nargs = 3, help = "PatchX index for convGMMs")
    parser.add_argument("--prev_layer_hwc", default = [-1, -1, -1],             type=int,   nargs = 3, help = "PatchX index for convGMMs")
    parser.add_argument("--filter_size",    default = [-1, -1],                 type=int,   nargs = 2, help = "PatchX index for convGMMs")
    parser.add_argument("--proto_size",     default = [-1, -1],                 type=int,   nargs = 2, help = "PatchX index for convGMMs")
    parser.add_argument("--clip_range",     default = [-100., 100.],            type=float, nargs = 2, help = "clip display to this range")
    parser.add_argument("--disp_range",     default = [-100., 100.],            type=float, nargs = 2, help = "clip display to this range")
    parser.add_argument("--pad",            default = 0.,                       type=float, help = "Overall padding")
    parser.add_argument("--w_pad",          default = 0.,                       type=float, help = "X padding")
    parser.add_argument("--h_pad",          default = 0.,                       type=float, help = "Y padding")

    parser.add_argument("--out",            required = True,                    type=str,   help="output file path")
    parser.add_argument("--mu_file",        default = "mus.npy",                type=str,   help="data points to plot for 2D visualisation")
    parser.add_argument("--sigma_file",     default = "sigmas.npy",             type=str,   help="data points to plot for 2D visualisation")
    parser.add_argument("--pi_file",        default = "pis.npy",                type=str,   help="data points to plot for 2D visualisation")

    parser.add_argument("--img_sequence",   default = False,                    type=eval,  help="flag if visualizing an image sequence")
    parser.add_argument("--sequence_path",  required = True,                    type=str,   help="path to sub-dirs containing visualization content")
    parser.add_argument("--epoch",          default = -1,                       type=int,   help="epoch to visualize protos for, if set to -1 (default) last epoch is used")

    FLAGS = parser.parse_args()

    if os.path.isabs(FLAGS.sequence_path) == False: log.error("--sequence_path must be absolute!")
    if os.path.isabs(FLAGS.out) == False: log.error("--out must be absolute!")

    if not os.path.exists(FLAGS.sequence_path): log.error("--sequence_path does not exist!")
    if not os.path.exists(FLAGS.out): os.makedirs(FLAGS.out)

    if FLAGS.img_sequence:
        if os.path.exists(FLAGS.sequence_path):                 # ./results/protos_T1/
            epoch_dirs = os.listdir(FLAGS.sequence_path)
            proto_img_list = [0] * len(epoch_dirs)
            for i in range(0, len(proto_img_list)-1):           # E_0 ... E_N, where N is num of training epochs
                sub_dir_path = FLAGS.sequence_path + f'/E{i}'
                if os.path.isdir(sub_dir_path) and os.path.exists(sub_dir_path):
                    prefix = os.path.join(sub_dir_path, FLAGS.prefix)
                    proto_img_list[i] = vis_img_data(prefix=FLAGS.prefix, prefix_path=prefix, suffix=i, **FLAGS)
            # IMAGEIO CREATE GIF
            images = list()
            for i in range(0, len(proto_img_list)):
                img_name = proto_img_list[i]
                if img_name.endswith('mus.png'):
                    images.append(imageio.imread(img_name))
            gif_path = os.path.join(FLAGS.out, '/protos.gif')
            imageio.mimwrite(gif_path, images, duration=1)
    else:
        if FLAGS.epoch == -1:
            epoch_dirs = os.listdir(FLAGS.sequence_path)
            all_epochs = sorted(int(x) for x in map(lambda epoch_desc: epoch_desc[1:], epoch_dirs))
            prefix = os.path.join(FLAGS.sequence_path, f'E{all_epochs[-1]}', FLAGS.prefix)
            vis_img_data(prefix_path=prefix, suffix=all_epochs[-1], **vars(FLAGS))
        else:
            if os.path.exists(FLAGS.sequence_path):
                prefix = os.path.join(FLAGS.sequence_path, FLAGS.prefix)
                vis_img_data(prefix_path=prefix, suffix=FLAGS.epoch, **FLAGS)
