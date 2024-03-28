EXP_ID="er-emnist-replay"

PROJ_PATH="${HOME}/git/AR"
DATA_DIR="${HOME}/datasets"
SAVE_DIR="${HOME}/exp-results/${EXP_ID}"

AR_MOD="cl_replay.architecture.ar"
export PYTHONPATH=$PYTHONPATH:$PROJ_PATH/src/cl_replay

LABEL_DIM=47

python3 -m cl_replay.architecture.rehearsal.experiment.Experiment_Rehearsal \
--project_name                  ICLR24                          \
--architecture                  ER-DNN                          \
--exp_group                     ER-DNN                          \
--exp_tags                      ER-DNN                          \
--exp_id                        "${EXP_ID}"                     \
--wandb_active                  no                              \
--dataset_dir                   "${DATA_DIR}"                   \
--dataset_load                  tfds                            \
--dataset_name                  emnist/balanced                 \
--renormalize01                 yes                             \
--np_shuffle                    yes                             \
--data_type                     32                              \
--num_tasks                     6                               \
--DAll                          0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
--T1                            5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 \
--T2                            0                               \
--T3                            1                               \
--T4                            2                               \
--T5                            3                               \
--T6                            4                               \
--num_classes                   "${LABEL_DIM}"                  \
--epochs                        100                             \
--batch_size                    128                             \
--log_level                     DEBUG                           \
--save_All                      yes                             \
--train_method                  fit                             \
--test_method                   eval                            \
--full_eval                     yes                             \
--single_class_test             no                              \
--model_type                    latent                          \
--callback_paths                ${AR_MOD}.callback              \
--global_callbacks              Log_Metrics                     \
--log_path                      "${SAVE_DIR}"                   \
--vis_path                      "${SAVE_DIR}"                   \
--ckpt_dir                      "${SAVE_DIR}"                   \
--replay_proportions            50. 50.                         \
--samples_to_generate           -1.                             \
--loss_coef                     off                             \
--budget_method                 class                           \
--storage_budget                2000                            \
--per_class_budget              50                              \
--per_task_budget               .005                            \
--rehearsal_type                batch                           \
--balance_type                  tasks             