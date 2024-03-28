import os, sys, re
os.environ["KERAS_BACKEND"] = "tensorflow"
import pprint

import numpy as np
import keras_cv
import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

from importlib import import_module
from matplotlib import pyplot

from tensorflow.keras import applications
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, Flatten, Conv2D, BatchNormalization, Normalization, RandomFlip, RandomRotation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

from cl_replay.api.parsing      import Kwarg_Parser, Command_Line_Parser
from cl_replay.api.utils        import log


class DatasetEncoder():
    ''' 
    This script encodes a specified tfds with an arbitrary (pre-trained) architecture. 
    The encoded dataset is then saved back to the file storage as numPy.
    
    Params:

    --encode_ds, --pretrain_ds, --finetune_ds
        * svhn_full
        * svhn_cropped
        * cifar10
        * cifar100
        * imagenette
    --encode_split, --pre-train_split, --finetune_split
        * train
        * train[:50%]
        * extra[20%:]
    --pretrain_epochs, --finetune_epochs
        * N
    --architecture:
        * VGG16
        * VGG19
        * resnet_v2.ResNet50V2
        * CustomEncoder
        * densenet.DenseNet201
    --weights
        * None
        * "imagenet"
    --pooling
        * None
        * "avg"
        * "max"
    --include_top: [yes, no]
    --output_layer
        * VGG16: block[1-5]_pool
        * ResNet50: conv[1-5]_block[1-6]_[0-3]_conv, post_relu
        * " " for last layer of encoder
    --bootstrep: [yes, no]
    --finetune: [0-N]
    --contrastive_learning: [yes, no]
    --split:
        * 10
        * 5 5
        * 2 2 2 2 2
        * 5 1 1 1 1 1
        * 10 10 10 10 10 10 10 10 10 10 

    Example Usage:

    1) SCL encoding 100% of SVHN train split, pre-trained on SVHN using 7% of extra split.

    python3 "${PROJ_PATH}/src/cl_replay/api/data/DatasetEncoder.py" \
        --encode_ds svhn_cropped \
        --encode_split train \
        --split 10 \
        --out_dir $HOME/datasets \
        --out_name svhn-7-ex \
        --architecture resnet_v2.ResNet50V2 \
        --pooling avg \
        --pretrain yes \
        --pretrain_ds svhn_cropped \
        --pretrain_split extra[:7%] \
        --pretrain_epochs 100 \
        --output_layer post_relu \
        --augment_data yes \
        --contrastive_learning yes \
        --contrastive_method supervised_npairs
    
    2) SCL encoding 50% of CIFAR-10 train split, pre-trained on CIFAR-10 using 50% of train split.

    python3 "${PROJ_PATH}/src/cl_replay/api/data/DatasetEncoder.py" \
        --encode_ds cifar10 \
        --encode_split train[50%:] \
        --split 10 \
        --out_dir $HOME/datasets \
        --out_name cifar10-50-ex \
        --architecture resnet_v2.ResNet50V2 \
        --pooling avg \
        --pretrain yes \
        --pretrain_ds svhn_cropped \
        --pretrain_split train[:50%] \
        --pretrain_epochs 100 \
        --output_layer post_relu \
        --augment_data yes \
        --contrastive_learning yes \
        --contrastive_method supervised_npairs
    '''
    def __init__(self, **kwargs):
        command_line_params = Command_Line_Parser().parse_args()
        self.parser         = Kwarg_Parser(external_arguments=command_line_params, verbose=True, **kwargs)

        self.dataset_load       = self.parser.add_argument('--dataset_load',    type=str,   default='tfds',     choices=['tfds', 'from_npz', 'construct', 'hdf5'], help='determine which API to use for loading a dataset file.')
        self.encode_ds          = self.parser.add_argument('--encode_ds',       type=str,   required=True,      help='tfds download string or name of a compressed pickle file.')
        if self.dataset_load == 'tfds':
            if self.encode_ds not in tfds.list_builders():
                log.error(f"--encode_ds must be an available dataset in {tfds.list_builders()}")
                sys.exit(0)
        self.encode_split       = self.parser.add_argument('--encode_split',    type=str,   default='train',    help='tfds split string to use for loading the encoding dataset e.g., "extra", "train[:20%]", "train[:1000]".')
        self.augment_data       = self.parser.add_argument('--augment_data',    type=str,   default='no',       choices=['yes', 'no'], help='perform random augmentation (norm, flip, rotate) before pre-training')
        if self.augment_data == 'yes': self.augment_data = True
        else: self.augment_data = False
        self.split              = self.parser.add_argument('--split',           type=int,   default=10,         help='how to split up dataset, e.g. [5,1,1,1,1,1] -> creates 6 datasets')
        self.batch_size         = self.parser.add_argument('--batch_size',      type=int,   default=128,        help='set the batch size for dataset processing.')
        self.as_supervised      = self.parser.add_argument('--as_supervised',   type=str,   default='yes',      choices=['yes', 'no'], help='flag for loading a tensorflow dataset with class labels.')
        if self.as_supervised == 'yes': self.as_supervised = True
        else: self.as_supervised = False
        if self.dataset_load != 'tfds':
            self.load_dir       = self.parser.add_argument('--load_dir',        type=str,   required=True,      help='set the directory to load datasets from if it is loaded from file storage.')
            if os.path.isabs(self.load_dir) == False:
                log.error("'--dataset_dir' must be an absolute path.")
                sys.exit(0)
            if not os.path.exists(self.load_dir): os.makedirs(self.load_dir)
        self.out_dir            = self.parser.add_argument('--out_dir',         type=str,   required=True,      help='set the default directory for saving encoded dataset files.')
        if os.path.isabs(self.out_dir) == False:
                log.error("'--out_dir' must be an absolute path.")
                sys.exit(0)
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
        self.out_name           = self.parser.add_argument('--out_name',        type=str,   help='filename of encoded dataset.')
        
        self.architecture       = self.parser.add_argument('--architecture',    type=str,   default='VGG16',    help='set the architecture to encode the dataset.')
        self.include_top        = self.parser.add_argument('--include_top',     type=str,   default='yes',      choices=['yes', 'no'], help='include top FC layers')
        if self.include_top == 'yes': self.include_top = True
        else: self.include_top = False
        self.output_layer       = self.parser.add_argument('--output_layer',    nargs='+',  type=str,   required=True,    help='sets the layers to extract features from (output layers).')
        self.weights            = self.parser.add_argument('--weights',         type=str,   default=None, choices=[None, 'imagenet'],   help='None: (random init.), "imagenet": pre-trained, requires ')      
        self.pooling            = self.parser.add_argument('--pooling',         type=str,   default='none', choices=['none', 'avg', 'max'], help='none: output of last conv block (4D tensor), avg: global avg. pooling (2D tensor), max: global max pooling' )
        if self.pooling == 'none': self.pooling = None
        
        self.pretrain           = self.parser.add_argument('--pretrain',        type=str,   default='no',       choices=['yes', 'no'], help='custom pre-training')
        self.pretrain_ds        = self.parser.add_argument('--pretrain_ds',     type=str,   choices=[None, 'svhn_cropped', 'cifar10', 'cifar100', 'imagenette'], help='see args "--encode_ds"')
        self.pretrain_split     = self.parser.add_argument('--pretrain_split',  type=str,   default='train',    help='tfds split string to use for loading the pre-training dataset e.g., "extra", "train[:20%]", "train[:1000]".')
        self.pretrain_epochs    = self.parser.add_argument('--pretrain_epochs', type=int,   default=50,         help='set top N layers to fine-tune on extraction DS')
        self.pretrain_classes   = self.parser.add_argument('--pretrain_classes',nargs='+',  type=int, help='pass classes as "0 1 2" to only pretrain on specific classes.')
        if self.pretrain == 'yes':
            self.pretrain = True
            if self.pretrain_ds == None:
                log.error("'--pretrain_ds' must be specified when '--pretrain' is set to 'yes'.")
                sys.exit(0)
            if self.pretrain_ds not in tfds.list_builders():
                log.error(f"--pretrain_ds must be an available dataset in {tfds.list_builders()}")
                sys.exit(0)
            self.weights = None
        else: self.pretrain = False
        
        self.finetune           = self.parser.add_argument('--finetune',        type=int,   default=0,          help='set top N layers to fine-tune on extraction DS')
        self.finetune_ds        = self.parser.add_argument('--finetune_ds',     type=str,   choices=[None, 'svhn_cropped', 'cifar10', 'cifar100'], help='see args "--encode_ds"')
        if self.finetune > 0:
            if self.finetune_ds not in tfds.list_builders():
                log.error(f"--finetune_ds must be an available dataset in {tfds.list_builders()}")
                sys.exit(0)
        self.finetune_split     = self.parser.add_argument('--finetune_split',  type=str,   default='train',    help='tfds split string to use for loading the fine-tuning dataset e.g., "extra", "train[:20%]", "train[:1000]".')
        self.finetune_epochs   = self.parser.add_argument('--finetune_epochs',type=int,   default=20,         help='set top N layers to fine-tune on extraction DS')

        self.bootstrap          = self.parser.add_argument('--bootstrap',       type=str,   default='yes',      choices=['yes', 'no'], help='set to "yes" if top structure shall be replaced by custom structure')
        if self.bootstrap == 'yes': self.bootstrap = True; self.include_top = False
        else: self.bootstrap = False
        
        self.contrastive_learning = self.parser.add_argument('--contrastive_learning', type=str,   default='no', choices=['yes', 'no'], help='set to "yes" for contrastive learning of encoder network')
        self.contrastive_method = self.parser.add_argument('--contrastive_method',  type=str, default='supervised_npairs', choices=['supervised_npairs', 'simclr'], help='set the constrative learning method.')
        if self.contrastive_learning == 'yes': self.contrastive_learning = True; self.bootstrap = False; self.include_top = False
        else: self.contrastive_learning = False

    # -----------------> DATA LOADING

    def load_via_tfds(self, ds_name, split_string):
        (xs, ys), info = tfds.load(
                ds_name,
                split=split_string,
                batch_size=-1, # loads full DS
                shuffle_files=False,
                as_supervised=self.as_supervised,
                with_info=True
        )
        self.h, self.w, self.c  = info.features['image'].shape
        self.num_classes = info.features['label'].num_classes
        c, _, counts = tf.unique_with_counts(x=ys)
        c_sorted = tf.argsort(c)
        counts_sorted = tf.gather(counts, c_sorted)
        c_sorted = tf.gather(c, c_sorted)

        print(
            #f'{info}\n',
            f'\nDS NAME/SPLIT:\t\t"{ds_name}"\t"{split_string}"\n',
            f'X/Y DIMS:\t\t{xs.shape}\t{ys.shape}\n',
            f'X/Y VALS (MIN/MAX):\t{tf.math.reduce_min(xs).numpy()}\t\t{tf.math.reduce_max(xs).numpy()}\n',
            f'LABEL DIST:\t\t{c_sorted}: {counts_sorted}\n',
        )

        return tfds.as_numpy(xs), tfds.as_numpy(ys)


    def load_from_npz(self, ds_name): #FIXME impl.
        d = np.load(open(os.path.join(self.load_dir, self.encode_ds),"rb"))
        print(d)
        if self.encode_ds.endswith("npz"):
            tr_x, tst_x , tr_y , tst_y = d.values() # a,b,c,d
            print("LOADED NPZ INFO ->", tr_x.shape, tst_x.shape, tr_y.shape, tst_y.shape, tr_x.max())
            return tr_x.astype(dt), tr_y.astype(dt), tst_x.astype(dt), tst_y.astype(dt)


    def load_from_dsk(self, ds_name): #FIXME impl.
        if os.path.isdir(ds_name):
            for f in os.path.listdir:
                if str(f).lower == 'training':
                    print(f)
                if str(f).lower == 'testing':
                    print(f)


    def prepare_tfds(self, ds_name, split_string):
        ds, info = (tfds.load(
                ds_name, data_dir="dataset", split=split_string, as_supervised=True)
                    .map(lambda image, _: image, num_parallel_calls=AUTOTUNE)
                    .shuffle(buffer_size=2 * self.batch_size)
                    .batch(self.batch_size, num_parallel_calls=AUTOTUNE)
                    .prefetch(AUTOTUNE)
        )
        return ds


    def load_dataset(self, dataset_name, train_split, pretrain=False):
        if self.dataset_load == 'tfds':
            self.raw_tr_xs, self.raw_tr_ys = self.load_via_tfds(dataset_name, train_split)
            self.raw_tst_xs, self.raw_tst_ys = self.load_via_tfds(dataset_name, "test") # FIXME: validation ???
        elif self.dataset_load == 'from_npz': #FIXME impl.
            self.raw_tr_xs, self.raw_tr_ys, self.raw_tst_xs, self.raw_tst_ys = self.load_from_npz(dataset_name)
        elif self.dataset_load == 'construct': #FIXME impl.
            self.raw_tr_xs, self.raw_tr_ys, self.raw_tst_xs, self.raw_tst_ys = self.load_from_npz(dataset_name)

        self.indices_train = np.arange(0, self.raw_tr_xs.shape[0])
        self.indices_test  = np.arange(0, self.raw_tst_xs.shape[0])

        if pretrain == True and self.pretrain_classes != None:
            if type(self.pretrain_classes) == int: self.pretrain_classes = [self.pretrain_classes]
            filtered_indices_tr, filtered_indices_tst = None, None
            
            for cls in self.pretrain_classes:
                concat_tr, concat_tst = self.get_class_indices(cls)
                if not np.any(filtered_indices_tr):
                    filtered_indices_tr, filtered_indices_tst = concat_tr, concat_tst
                else:
                    filtered_indices_tr = np.concatenate((filtered_indices_tr, concat_tr))
                    filtered_indices_tst = np.concatenate((filtered_indices_tst, concat_tst))
            # filter DS
            self.raw_tr_xs, self.raw_tr_ys = self.raw_tr_xs[filtered_indices_tr], self.raw_tr_ys[filtered_indices_tr]
            self.raw_tst_xs, self.raw_tst_ys = self.raw_tst_xs[filtered_indices_tst], self.raw_tst_ys[filtered_indices_tst]

        print(self.raw_tr_xs.shape, self.raw_tr_ys.shape)
        print(self.raw_tst_xs.shape, self.raw_tst_ys.shape)

        # flatten & one-hot encode
        onehot_raw_tr_ys = self.raw_tr_ys.reshape(-1)
        onehot_raw_tst_ys = self.raw_tst_ys.reshape(-1)

        self.onehot_raw_tr_ys = np.eye(self.num_classes)[onehot_raw_tr_ys]
        self.onehot_raw_tst_ys = np.eye(self.num_classes)[onehot_raw_tst_ys]

        if (pretrain == False) and (self.weights == 'imagenet'): # convert to BGR where each color channel is zero-centered
            print("\nPREPROCESSING DATA, SINCE IMAGENET WEIGHTS WERE LOADED...\n")
            self.raw_tr_xs  = preprocess_input(self.raw_tr_xs)
            self.raw_tst_xs = preprocess_input(self.raw_tst_xs)

        self.input_shape = self.h, self.w, self.c


    def get_class_indices(self, classes):
        ''' Returns indices of the data for specific classes. ''' 
        int_class   = int(classes)
        mask_train  = (self.raw_tr_ys == int_class)
        mask_test   = (self.raw_tst_ys == int_class)

        return self.indices_train[mask_train], self.indices_test[mask_test]

    # -----------------> VISUALIZATION

    def plot_filters(self, model):
        for layer in model.layers:
            if 'conv' in layer.name and layer.trainable == True:
                if len(layer.get_weights()) == 2:
                    filters, biases = layer.get_weights()
                    print(layer.name, layer.trainable, filters.shape)


    def visualize_input(self, xs, ys, num_images):
        """ Visualize num_images of the data (xs=samples, ys=labels). """
        num_images, num_cols, num_rows = 9, 3, 3
        fig, axes = pyplot.subplots(num_rows, num_cols, figsize=(num_cols, num_rows))
        for i in range(num_images):
            if num_rows != 1:
                ax = axes[i//num_cols, i%num_cols]
            else:
                ax = axes[i%num_cols]
            ax.imshow(xs[i], cmap='gray') # cmap='gray_r' for color swap

            ax.set_xticklabels([])
            ax.set_yticklabels([])

            ax.set_aspect('equal')
            ax.set_title('Label: {}'.format(np.argmax(ys[i])))
            pyplot.subplots_adjust(wspace=0.1, hspace=0.1)
            pyplot.axis('off')
        pyplot.show()

    
    def visualize_filters(self, model, n_filters, layer_idx):
        """ Visualize first N filters of i-th conv-layer (credits to MrNouman). """
        filters, biases = model.layers[layer_idx].get_weights()
        filter_min, filter_max = filters.min(), filters.max()
        filters = (filters - filter_min) / (filter_max - filter_min) # normalize [0,1]
        ix = 1 
        for i in range(n_filters):
            f = filters[:, :, :, i]
            for j in range(self.c):
                ax = pyplot.subplot(n_filters, self.c, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                pyplot.imshow(f[:, :, j])
                ix += 1
        pyplot.show()


    def visualize_fmaps(self, base_model, block_indices):
        """ Visualize feature maps derived from conv-blocks (credits to MrNouman).
            Arguments need to be integers of last conv-layer per visualized block.
        """
        outputs = [base_model.layers[i].output for i in block_indices]
        model = Model(inputs=base_model.input, outputs=outputs)
    
        rnd_img = self.raw_tr_xs[np.random.randint(0, self.raw_tr_xs.shape[0])]
        rnd_img = np.expand_dims(rnd_img, axis=0) # reshape to (1,H,W,C)
        if self.weights == 'imagenet':
            rnd_img = preprocess_input(rnd_img)

        feature_maps = model.predict(rnd_img)

        sqr = 8
        for fmap in feature_maps:
            ix = 1
            for _ in range(sqr):
                for _ in range(sqr):
                    ax = pyplot.subplot(sqr, sqr, ix)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    pyplot.imshow(fmap[0, :, :, ix-1]) #cmap='gray'
                    ix += 1
            pyplot.show()

    # -----------------> MODEL CREATION/LOADING/FIT&EVAL/WEIGHT TRANSFER
        
    def create_base_model(self, create_model=True, augment=False):
        try:
            print(f"\nBUILDING {self.architecture} BASE MODEL...\n")
            if self.architecture == 'CustomEncoder':
                self.base_model = self.build_custom_encoder(self.input_shape)
            else:
                if self.architecture.find('.') != -1:
                    sp = re.split('\.', self.architecture)
                    self.architecture = sp[1]
                    pkg = import_module(f'tensorflow.keras.applications.{sp[0]}')
                else:
                    pkg = import_module('tensorflow.keras.applications')

                network_module = getattr(pkg, self.architecture)
                model_kwargs = {
                    "include_top"   : False if self.bootstrap == 'yes' else self.include_top, # omit top FC layers
                    "input_tensor"  : None,
                    "input_shape"   : self.input_shape, # specify based on data shape, however, no smaller than (32,32) allowed
                    "weights"       : self.weights,     # pre-trained on ImageNet
                    "pooling"       : self.pooling,     # specify pooling for last conv-block
                }
                print(f"\nMODEL KWARGS:\n")
                pprint.pprint(model_kwargs)

                self.base_model = getattr(pkg, self.architecture)(**model_kwargs)
                self.base_in = self.base_model.layers[0].input
                self.base_out = self.base_model.layers[-1].output

            # model is created such that it can be trained end-to-end with one output layer
            if augment == True: # perform a random image augmentation for each sample
                if self.contrastive_method == 'supervised_npairs':
                    if not self.architecture == 'CustomEncoder':
                        self.base_in = Input(shape=self.input_shape)
                    norm = Normalization()
                    norm.adapt(self.raw_tr_xs)
                    norm = norm(self.base_in)
                    rnd_flip = RandomFlip("horizontal")(norm)
                    rot = RandomRotation(0.02)(rnd_flip)
                    self.encoder = self.base_model(rot)
                if self.contrastive_method == 'simclr':
                    self.encoder = self.base_out

            # prepare output from all layers we want to extract information from!
            self.model_outputs = []
            trainables = [(idx, l.name) for idx, l in enumerate(self.base_model.layers) if l.trainable == True]
            if type(self.output_layer) == str: self.output_layer = [self.output_layer]  # list wrap
            for ol in self.output_layer: # find trainables to include as output
                for item in trainables:
                    if ol in item: self.model_outputs.append(self.base_model.layers[item[0]].output)
            #print("\nTRAINABLE LAYERS:\n")
            #pprint.pprint(trainables)
            print("\nFINAL (FE) MODEL OUTPUT:\n")
            pprint.pprint(self.model_outputs)
            # -----------------> BOOTSTRAP
            if self.bootstrap == True: # bootstrap network with a new top structure
                flat            = Flatten(name="flatten")(self.base_out)
                dense_1         = Dense(name="dense_1", units=2048, activation="relu")(flat)
                drop_1          = Dropout(0.3)(dense_1)
                dense_2         = Dense(name="dense_2", units=1024, activation="relu")(drop_1)
                drop_2          = Dropout(0.3)(dense_2)
                self.base_out   = Dense(name="prediction", units=self.num_classes, activation="softmax")(drop_2)

            if self.contrastive_learning == True:
                projection_units = 128

                if self.contrastive_method == 'supervised_npairs':
                    in_ = self.encoder
                    # adds a projection head, a MLP with a dense layer of 2048 with relu activation & an output vector of 128 projection units
                    if self.architecture == 'CustomEncoder': # flatten input
                        in_           = Flatten(name="flatten")(self.encoder)
                    dense_1 = Dense(2048, activation="relu")(in_)
                    self.base_out = Dense(projection_units, activation="relu")(dense_1)
            
                if self.contrastive_method == 'simclr':
                    CONTRASTIVE_AUGMENTATION = {
                        "crop_area_factor": (0.08, 1.0),
                        "aspect_ratio_factor": (3 / 4, 4 / 3),
                        "color_jitter_rate": 0.8,
                        "brightness_factor": 0.2,
                        "contrast_factor": 0.8,
                        "saturation_factor": (0.3, 0.7),
                        "hue_factor": 0.2,
                    }
                    augmenter = self.get_simclr_augmenter(self.input_shape, **CONTRASTIVE_AUGMENTATION)

                    self.model = SimCLRTrainer(
                        encoder=self.base_model,
                        augmenter=augmenter,
                        projector=Sequential(
                        [
                            Dense(projection_units, activation="relu"),
                            Dense(projection_units),
                            BatchNormalization(),
                        ],
                        name="projector",
                        )
                    )

            if create_model:
                # specifies input & output layer for feature extraction (name of model layer)    
                self.model = Model(inputs=self.base_in, outputs=self.base_out, name="encoder-network") 
                self.model.summary()

            return self.base_model
            
        except Exception as ex:
            import traceback
            log.error(traceback.format_exc())
            log.error(f'ERROR WHILE LOADING MODEL "{self.architecture}": {ex}.')


    def build_custom_encoder(self, input_dims):
        """ builds a 5-layerered CNN encoder network, as presented in Gido M. van de Ven (2020) """
        self.base_in = Input(input_dims)
        # BLOCK 1
        val = Conv2D(16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(self.base_in) # padding='same'
        val = BatchNormalization()(val)
        # BLOCK 2
        val = Conv2D(32, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(val) 
        val = BatchNormalization()(val)
        # BLOCK 3
        val = Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(val) 
        val = BatchNormalization()(val)
        # BLOCK 4
        val = Conv2D(128, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(val) 
        val = BatchNormalization()(val)
        # BLOCK 5
        val = Conv2D(256, kernel_size=(3, 3), strides=2, padding='same', activation='relu')(val) 
        self.base_out = BatchNormalization()(val)

        return Model(inputs=self.base_in, outputs=self.base_out, name='encoder_network')
    

    def fit_model(self, model, fine_tune=False, epochs=100):
        
        if self.contrastive_learning == True:
            learning_rate = 0.001
            opt = Adam(learning_rate)
            metrics = None
            
            # contrastive loss needs scalar repr. of labels
            ys_tr = self.raw_tr_ys
            ys_tst = self.raw_tst_ys

            if self.contrastive_method == 'supervised_npairs':
                temperature = 0.05
                loss = SupervisedContrastiveLoss(temperature)
                model.compile(   # FIXME: this stuff is a mess...
                    optimizer=opt, 
                    loss=loss,
                    metrics=metrics,
                )

            if self.contrastive_method == 'simclr':
                temperature = 0.1
                if fine_tune == True:
                    loss = tf.keras.losses.sparse_categorical_crossentropy
                    model.compile(
                        optimizer=opt,
                        loss=loss,
                        metrics=['accuracy']
                    )
                else: 
                    loss = keras_cv.losses.SimCLRLoss(temperature) # takes two unlabeled projections
                    model.compile(  # FIXME: this stuff is a mess...
                        encoder_optimizer=opt, 
                        encoder_loss=loss,
                    )
        else:
            learning_rate = 0.0001 # rather pick a small learning rate!
            opt = Adam(learning_rate)
            loss = tf.keras.losses.categorical_crossentropy # if non one-hot: use "tf.keras.losses.sparse_categorical_crossentropy"
            metrics=['accuracy']

            model.compile(   # FIXME: this stuff is a mess...
                optimizer=opt, 
                loss=loss,
                metrics=metrics,
            )

            ys_tr = self.onehot_raw_tr_ys
            ys_tst = self.onehot_raw_tst_ys

        print("\nFITTING MODEL...\n")
        if self.contrastive_learning == True and self.contrastive_method == 'simclr':
            if fine_tune == False:  # self-supervised
                batch_size = None
                steps_per_epoch = None

                data = tf.data.Dataset.from_tensor_slices((self.raw_tr_xs, ys_tr))
                data = data.map(lambda image, _: image, num_parallel_calls=tf.data.AUTOTUNE)
                data = data.shuffle(buffer_size=2 * self.batch_size)
                data = data.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
                data = data.prefetch(tf.data.AUTOTUNE)
                
                xs = data
                ys = None
            else:                   # fine-tuning
                batch_size = 128
                steps_per_epoch = (self.raw_tr_xs.shape[0]//self.batch_size)
                xs = self.raw_tr_xs
                ys = ys_tr
        else:
            batch_size = self.batch_size
            steps_per_epoch = (self.raw_tr_xs.shape[0]//self.batch_size)
            xs = self.raw_tr_xs
            ys = ys_tr

        model.fit(
            x = xs,
            y = ys, 
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
        )
        
        if not (self.contrastive_method == 'simclr' and fine_tune == True): # FIXME: simclr has no test_step implemented!
            pass
        else:
            print("\nEVALUATING MODEL...\n")
            model.evaluate(x=self.raw_tst_xs, y=ys_tst)


    def get_simclr_augmenter(
        self,
        input_shape,
        crop_area_factor,
        aspect_ratio_factor,
        color_jitter_rate,
        brightness_factor,
        contrast_factor,
        saturation_factor,
        hue_factor,
    ):
        return Sequential(
            [
                Input(shape=input_shape),
                keras_cv.layers.Rescaling(scale=1.0 / 255), # [0,1] scaling
                keras_cv.layers.RandomFlip("horizontal"),
                keras_cv.layers.RandomCropAndResize(
                    target_size=(input_shape[0], input_shape[1]),
                    crop_area_factor=crop_area_factor,
                    aspect_ratio_factor=aspect_ratio_factor,
                ),
                keras_cv.layers.RandomApply(
                    keras_cv.layers.RandomColorJitter(
                        value_range=(0, 1),
                        brightness_factor=brightness_factor,
                        contrast_factor=contrast_factor,
                        saturation_factor=saturation_factor,
                        hue_factor=hue_factor,
                    ),
                    rate=color_jitter_rate,
                ),
            ]
        )


    def extract_weights(self, model):
        print("\nEXTRACTING WEIGHTS FROM TRAINED MODEL...\n")
        conv_weights = []
        if model.layers:
            for component in model.layers:
                if isinstance(component, Model):
                    for layer in component.layers:
                        if component.trainable == True and 'conv' in component.name:
                            print(f'saving weights of {layer.name}...')

                if isinstance(component, Layer):
                    if component.trainable == True and 'conv' in component.name:
                        print(f'saving weights of {component.name}...')
                        conv_weights.append(component.get_weights())
        return conv_weights


    def set_weights(self, model, conv_weights):
        print("\nTRANSFERING WEIGHTS TO FRESH MODEL...\n")
        ix = 0
        for layer in model.layers:
            if 'conv' in layer.name and layer.trainable == True:
                print(f'setting weights of {ix}: {layer.name}...')
                layer.set_weights(conv_weights[ix])
                ix += 1


    def print_weights(self, model):
        for layer in model.layers:
            if layer.trainable == True and 'conv' in layer.name:
                print(layer.name, ":")
                for weight in layer.get_weights():
                    print("\t", weight.mean())

    # -----------------> EXTRACTION: SPLIT & ENCODE
    
    def check_split(self):
        """ Prepare instances for separate data partitions. """
        self.last_class = -1
        if type(self.split) == int:
            if self.split != self.num_classes:
                log.error("--split was adjusted to match the total number of classes for the current dataset.")
                self.split = self.num_classes
            self.split = [self.split]
        elif type(self.split) == list:
            self.split = np.array(self.split)
            if self.split.sum() != self.num_classes:
                log.error(f"defined --split proportions do not match the total number of classes, expected a sum of: {self.num_classes}.")
                sys.exit(0)
        self.tr_indices = list()
        self.tst_indices = list()
        for partition in self.split:
            partition_tr_indices, partition_tst_indices = None, None
            classes = np.arange((self.last_class+1), (self.last_class+partition)+1)
            for cls in classes:
                int_class = int(cls)
                mask_train = (self.raw_tr_ys == int_class)
                mask_test = (self.raw_tst_ys == int_class)
                if partition_tr_indices is None: # Nothing added yet (init.)
                    partition_tr_indices = self.indices_train[mask_train]
                    partition_tst_indices = self.indices_test[mask_test]
                else: # Concat to existing indices
                    partition_tr_indices = np.concatenate((partition_tr_indices, self.indices_train[mask_train]), axis=0)
                    partition_tst_indices = np.concatenate((partition_tst_indices, self.indices_test[mask_test]), axis=0)
                self.last_class = cls
            self.tr_indices.append(partition_tr_indices)
            self.tst_indices.append(partition_tst_indices)
    

    def encode_data(self, model, iter, idx_partition, train):
        """ Encodes pre-defined data split batches with the selected network. 
            Each split is encoded separately & saved as an individual dataset-file to avoid OOM issues for high-dims data.
        """ 
        if train == True: 
            raw_data_xs = self.raw_tr_xs
            if self.contrastive_learning == True:
                raw_data_ys = self.raw_tr_ys
            else: raw_data_ys = self.onehot_raw_tr_ys
        else: 
            raw_data_xs = self.raw_tst_xs
            if self.contrastive_learning == True:
                raw_data_ys = self.raw_tst_ys
            else: raw_data_ys = self.onehot_raw_tst_ys
        # encode batch-by-batch
        encoded_xs = None; encoded_ys = None
        for step in range(iter):
            lower = int(step*self.batch_size)
            upper = int((step+1)*self.batch_size)
            indices = idx_partition[lower:upper]
            xs = raw_data_xs[indices]
            ys = raw_data_ys[indices]

            if self.contrastive_learning == True:
                # one-hot encode in case of SCL
                ys = ys.reshape(-1)
                ys = np.eye(self.num_classes)[ys]

            # encoding & reshaping in case of 2D output
            xs = np.array(model(xs, training=False))
            if len(xs.shape) == 2:
                xs = np.reshape(xs, (self.batch_size, 1, 1, xs.shape[1]))

            if type(xs) == list:
                raise NotImplementedError("Multiple outputs not supported yet! Please specify a single output layer.")
                sys.exit(-1)
                for i, tensor in enumerate(xs):
                    xs[i] = tensor.numpy()
            
            # concat
            if encoded_xs is None:
                encoded_xs = xs
                encoded_ys = ys
            else:
                encoded_xs = np.concatenate((encoded_xs, xs), axis=0)
                encoded_ys = np.concatenate((encoded_ys, ys), axis=0)
        return encoded_xs, encoded_ys


    def main(self):
        base_model = None
        if self.pretrain == True:
            # -----------------> DATA LOADING 1: PRE-TRAIN
            self.load_dataset(self.pretrain_ds, self.pretrain_split, self.pretrain)
            #self.visualize_input(self.raw_tr_xs[:9], self.onehot_raw_tr_ys[:9], num_images=9)
            
            # -----------------> MODEL CREATION 1 (PRE-TRAINING)
            if self.contrastive_learning and self.contrastive_method == 'simclr':
                create_model = False 
            else:
                create_model = True
            
            base_model = self.create_base_model(create_model=create_model, augment=self.augment_data)
            #self.plot_filters(self.model)

            # -----------------> CUSTOM PRE-TRAINING
            print("\nPRE-TRAINING...\n")
            self.fit_model(self.model, epochs=self.pretrain_epochs)
            
            #self.print_weights(self.model.encoder)
            #print("\nVISUALIZE FILTERS...\n")
            #self.visualize_filters(self.model, 6, 2)
            #print("\nVISUALIZE FEATURE MAPS ON RANDOM IMAGE...\n")
            #self.visualize_fmaps(base_model, [2, 5, 9, 13, 17])

        # -----------------> DATA LOADING 2: FINE-TUNE
        if self.finetune > 0:
            self.load_dataset(self.finetune_ds, self.finetune_split)
            #self.visualize_input(self.raw_tr_xs[:9], self.onehot_raw_tr_ys[:9], num_images=9)

            if base_model == None:
                # -----------------> MODEL CREATION 2 (FINE-TUNE)
                base_model = self.create_base_model(augment=self.augment_data, create_model=True)
                #self.plot_filters(self.model)
            if self.contrastive_learning == True and self.contrastive_method == 'simclr': # special case, since augmentation pipeline differs and projector changes...
                CLASSIFICATION_AUGMENTATION = {
                    "crop_area_factor": (0.8, 1.0),
                    "aspect_ratio_factor": (3 / 4, 4 / 3),
                    "color_jitter_rate": 0.05,
                    "brightness_factor": 0.1,
                    "contrast_factor": 0.1,
                    "saturation_factor": (0.1, 0.1),
                    "hue_factor": 0.2,
                }
                augmenter = self.get_simclr_augmenter(self.input_shape, **CLASSIFICATION_AUGMENTATION)
                
                self.model = Sequential(
                    [
                        Input(shape=self.input_shape),
                        augmenter,
                        self.base_model,
                        Dense(self.num_classes),
                    ],
                    name="finetuning_model",
                )
            # -----------------> FINE-TUNE
            if (self.bootstrap == True) or (self.include_top == True):      # only freeze layers in case of default top structure or custom bootstrap
                for l in base_model.layers[:-self.finetune]: # freeze layers
                    l.trainable = False
                    #print(f"Freezing {l.name}!")
            
            print("\nFINE-TUNING...\n")
            self.fit_model(self.model, fine_tune=True, epochs=self.finetune_epochs)

        # -----------------> WEIGHT TRANSFER
        """
        if (self.pretrain == True) or (self.finetune > 0):
                conv_weights = self.extract_weights(self.model)            
                # reset model input-output pipeline back to it's origin after fine-tuning and set outputs for FE      
                self.set_weights(self.model, conv_weights)
                self.visualize_filters(self.model, 6, 2)
                self.visualize_fmaps(base_model, [2, 5, 9, 13, 17])
        """

        # -----------------> DATA LOADING 3: EXTRACT
        self.load_dataset(self.encode_ds, self.encode_split)
        #self.visualize_input(self.raw_tr_xs[:9], self.onehot_raw_tr_ys[:9], num_images=9)
        #self.print_weights(self.model)

        if base_model == None:
            # -----------------> MODEL CREATION 3 (NO PRE-TRAIN & FINE-TUNE)
            base_model = self.create_base_model(create_model=False, augment=False) # postpone creation
        
        # readjust input-output pipeline for feature-extraction (after re-training & fine-tuning)
        if self.contrastive_learning == True:  
            out = self.encoder                  #TODO: fix problem where we can not target a specific layer in the encoder network for SCL (amix of augmentation layers and a functional object)
        else:                                   # weight transfer is not possible by re-adjusting layer connectivity, would need to serialize parts of the model and target specific layers
            out = self.model_outputs            # try to save/load model weights with skip_mismatch=True when using model.load_weights()
        self.model = Model(inputs=self.base_in, outputs=out, name="encoder-network")
        self.model.summary(show_trainable=True)

        #self.print_weights(self.model) # check model weights
        #self.plot_filters(self.model)

        # -----------------> ENCODE & SAVE DATA
        print("\nENCODING DATASET...\n")
        self.check_split()
        for i, (xs_partition_tr, xs_partition_tst) in enumerate(zip(self.tr_indices, self.tst_indices)): # np arrays holding indices
            # calculate training iterations with drop remainder
            xs_tr_iter = xs_partition_tr.shape[0] // self.batch_size
            xs_tst_iter = xs_partition_tst.shape[0] // self.batch_size
            
            tr_x, tr_y = self.encode_data(self.model, xs_tr_iter, xs_partition_tr, train=True)
            tst_x, tst_y = self.encode_data(self.model, xs_tst_iter, xs_partition_tst, train=False)

            print(
                f"\nENCODED PARTITION: {i}",
                f"\n\tPARTITION DATA:",
                f"\n\t\tTRAIN:\t{tr_x.shape}, {tr_y.shape}",
                f"\n\t\tTEST:\t{tst_x.shape}, {tst_y.shape}",
                f"\n\tPARTITION DATA:",
                f"\n\t\tTRAIN (MIN/MAX): {tr_x.min()}/{tr_x.max()}",
                f"\n\t\tTEST (MIN/MAX): {tst_x.min()}/{tst_x.max()}",
                f"\n\tPARTITION CLASSES:",
                f"\n\tTRAIN\t{np.unique(np.argmax(tr_y, axis=1), return_counts=True)[1]}",
                f"\n\tTEST\t{np.unique(np.argmax(tst_y, axis=1), return_counts=True)[1]}",
            )
            if self.out_name == None:
                self.out_name = f"{self.architecture}_encoded-{self.encode_ds}-partition{i}"
            partition_filename = f'{self.out_dir}/{self.out_name}.npz'
            
            np.savez(partition_filename, a=tr_x, b=tst_x, c=tr_y, d=tst_y)
            print(f"\nSAVED ENCODED DATASET TO: {partition_filename}")


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)


class SimCLRTrainer(keras_cv.training.ContrastiveTrainer):
    def __init__(self, encoder, augmenter, projector, probe=None, **kwargs):
        super().__init__(
            encoder=encoder,
            augmenter=augmenter,
            projector=projector,
            probe=probe,
            **kwargs,
        )


if __name__ == '__main__':
    dse = DatasetEncoder()
    dse.main()
