# Info
This is the code repository for "Adiabatic replay for continual learning" and "An analysis of best-practice strategies for replay and rehearsal in continual learning". It contains a self-written ML framework to facilitate the experimental evaluation of various CL strategies using TensorFlow, Keras and NumPy, as well as an instruction to reconstruct the experiments from these articles.

The full-text articles can be found online at: 
TBA LINK 1
TBA LINK 2

**"Adiabatic replay for continual learning" (IJCNN 2024)**:
The paper introduces a generative replay-based approach for CL termed adiabatic replay (AR), which achieves CL in constant time and memory complexity by making use of the (very common) situation where each new learning phase is adiabatic, i.e., represents only a small addition to existing knowledge. AR is evaluated on some common task-splits for MNIST, Fashion-MNIST, E-MNIST, and a feature encoded version of SVHN and CIFAR-10.

**"An analysis of best-practice strategies for replay and rehearsal in continual learning" (CVPR - CLVISION WORKSHOP 2024)**:
This study evaluates the impact of different choices in the context of class-incremental continual learning using replay. The investigation focuses on experience replay, generative replay using either VAEs or GANs and more general replay-strategies like loss-weighting. It is evaluated on a broad variety of task-splits for MNIST, Fashion-MNIST, E-MNIST and a latent version of SVHN and CIFAR-10. 

## Citation
If you use this code as a basis for your own research, please cite the paper:
```
@misc{krawczyk2024adiabatic,
      TBA
}
```
```
@misc{krawczyk2024empirical,
      TBA
}
```

## Note
If you encounter any bugs, unexpected behavior, have open questions or simply need support with the shipped source code - feel free to contact me via mail: _Alexander.Krawczyk@cs.hs-fulda_

## Installation & requirements
The code was tested on a machine with the Ubuntu 22.04.4 LTS OS, running with Python 3.10.12 and using the python packages listed in the `requirements.txt`. GPU support is enabled for NVIDIA graphics cards using CUDA 11.8 and CUDNN 8.6.0.

## Running experiments
The experiments for AR, DGR and ER can be run with a single command, e.g., `bash /path/to/AR/bash/dgr/mnist.bash`. Please check the environment variables and arguments for the python parser in the corresponding bash file and adjust these to your needs to freely experiment with different parameter settings. By running an experiment, the resulting metrics (.csv) are saved to the file system by default.

### To generate the latent datasets for SVHN and CIFAR-10:
In order to create the latent version of SVHN and CIFAR-10 as used in the experiments, please run these commands first. This will run the contrastive training of an extractor network and create an .npz file with the encoded datasets at the end of the process:
```
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
```
```
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
```
### Some settings to consider when manipulating the shipped .bash files:
Feel free to experiment with different settings or simply reconstruct some of the experiments from the articles.
1) Data settings: \
```--dataset_name: ["mnist", "fashion_mnist", "emnist/balanced"] ```: \
Load a dataset via TensorFlow datasets (tfds) API. \
```--dataset_name: ["svhn-7-ex.npz", "cifar10-50-ex.npz"]```: \
Load a dataset via numPy. \
```--dataset_load: ["tfds", "from_npz"] ```: \
Use "tfds" when loading MNIST, Fashion-MNIST and E-MNIST. Use "from_npz" when using your own (.npz/.npy) datasets or latent-SVHN and CIFAR-10.

2) Task settings: \
```--DAll: 0 1 2 3 4 5 6 7 8 9 ```: \
This list should contain all classes for the joint training- and test datasets. \
```--num_tasks: 4 ```: \
Set this to the number of total task identifiers "--TX". \
```--TX: 4 2```: \
Specify the task sequence in this manner, each task identifier is followed by the classes contained in this task. 

3) Model-specific settings: 
    1) DGR & ER loss coefficients: \
    ```--loss_coef: ["off", "class_balanced", "task_balanced"]```: \
    This defines the loss weighting strategy. 
    2) DGR specifics: \
    ```--replay_proportions: -1. -1 --samples_to_generate: 1.```: \
    These settings are used for the balanced replay strategy. \
    ```--replay_proportions: 50. 50. --samples_to_generate: -1.```: \
    These settings are used for the constant replay strategy. \
    ```--drop_generator: ["no", "yes"]```: \
    To drop the generator after each task. \
    ```--drop_solver: ["no", "yes"]```: \
    To drop the solver after each task. 
    
    3) ER specific: \
    ```--storage_budget: 1500 --per_class_budget: 50```: \
    This adjusts the buffer capacity and how many samples of each class are saved after each consecutive sub-task.

### Visualize Gaussian densities (AR):
To visualize the GMM prototypes after training you can run the following script to generate a .png image based on the learned Gaussien densities. Please adjust the `--sequence_path` and `--out` arguments as required. 
```
python3 "${PROJ_PATH}/src/cl_replay/api/utils/plot/vis_protos.py"       \
    --sequence_path "${SAVE_DIR}/protos/${EXP_ID}_protos_T1"            \
    --prefix "${EXP_ID}_L2_GMM_"                                        \
    --out "${SAVE_DIR}/protos/visualized/protos_T1"                     \
    --epoch -1                                                          \
    --channels 1                                                        \
    --proto_size 28 28                                                  \
    --pad   " 0.1 "                                                     \
    --h_pad " 0."                                                       \
    --w_pad " -10."                                                     ;
```
