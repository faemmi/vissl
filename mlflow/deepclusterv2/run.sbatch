#!/bin/bash -x
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=06:00:00
#SBATCH --output=/p/home/jusers/emmerich1/juwels/job-logs/%j-stdout.txt
#SBATCH --error=/p/home/jusers/emmerich1/juwels/job-logs/%j-stderr.txt

NODES=${NODES:-1}
GPUS=${GPUS:-4}
EPOCHS=${EPOCHS:-1}
DATASET=${DATASET:-daily}
CHECKPOINT_FOLDER_ID=${CHECKPOINT_FOLDER_ID:-$SLURM_JOB_ID}
LOG_LEVEL=${LOG_LEVEL:-INFO}
TRACK_TO_MANTIK=${TRACK_TO_MANTIK:-True}

if [ "$DATASET" = "hourly" ]; then
    DATASET="[remote_dataset_jsc_hourly]"
    N_SAMPLES="35064";
else 
    DATASET="[remote_dataset_jsc_daily]"
    N_SAMPLES="1461";
fi

if [ -z "$CPUS" ]; then
    echo "Using ${GPUS} GPUs per node";
else
    echo "Using ${CPUS} CPUs per node"
    CPU_CONFIG_ARGS='
        config.MACHINE.DEVICE=cpu
        config.DISTRIBUTED.BACKEND=gloo
        config.OPTIMIZER.use_larc=False
        config.MODEL.AMP_PARAMS.USE_AMP=False
    '
    export CUDA_VISIBLE_DEVICES="";
fi

if [ "$NODES" -gt "1" ]; then
    RUN_ID=${SLURMD_NODENAME}i:29500;
else
    RUN_ID=auto;
fi

if [ "$EPOCHS" -gt "999" ]; then
    LENGTHS="[0.01,0.99]";
elif [ "$EPOCHS" -gt "799" ]; then
    LENGTHS="[0.0125,0.9875]";
elif [ "$EPOCHS" -gt "599" ]; then
    LENGTHS="[0.0166667,0.9833333]";
elif [ "$EPOCHS" -gt "499" ]; then
    LENGTHS="[0.02,0.98]";
elif [ "$EPOCHS" -gt "399" ]; then
    LENGTHS="[0.025,0.975]";
elif [ "$EPOCHS" -gt "101" ]; then
    LENGTHS="[0.05,0.95]";
elif [ "$EPOCHS" -gt "5" ]; then
    LENGTHS="[0.1,0.9]";
elif [ "$EPOCHS" -gt "2" ]; then
    LENGTHS="[0.002563,0.997437]";
elif [ "$EPOCHS" -gt "1" ]; then
    LENGTHS="[0.00641,0.99359]";
else
    LENGTHS="[0.0128,0.9872]";
fi

echo "Submitting VISSL"
echo "NODES=${SLURM_NNODES}"
echo "N_CPUS=${SLURM_CPUS_ON_NODE}"
echo "GPUS=${SLURM_GPUS_PER_NODE}"
echo "EPOCHS=${EPOCHS}"
echo "LENGTHS=${LENGTHS}"
echo "RUN_ID=${RUN_ID}"
echo "DATASET=${DATASET}"
echo "N_SAMPLES=${N_SAMPLES}"
echo "CPU_CONFIG_ARGS=${CPU_CONFIG_ARGS}"
echo "LOG_LEVEL=${LOG_LEVEL}"

REPO_DIR=/p/home/jusers/emmerich1/juwels/code/vissl

BASE_DIR=/p/scratch/${SLURM_JOB_ACCOUNT}/maelstrom/${SLURM_JOB_USER}/deepclusterv2/${CHECKPOINT_FOLDER_ID}
CHECKPOINTS_DIR=${BASE_DIR}/checkpoints
TENSORBOARD_LOG_DIR=${BASE_DIR}/tensorboard-logs
TMP_DATA_DIR=${BASE_DIR}/tmp
CLUSTERING_DIR=${BASE_DIR}/clustering

mkdir -p ${BASE_DIR} ${CHECKPOINTS_DIR} ${TENSORBOARD_LOG_DIR} ${TMP_DATA_DIR} ${CLUSTERING_DIR}

echo "Run directory created at ${BASE_DIR}"

if [ $TRACK_TO_MANTIK == "True" ]; then
    source /p/project/deepacf/maelstrom/emmerich1/venvs/mantik/bin/activate
    source /p/home/jusers/emmerich1/juwels/code/a6/mantik.env > /dev/null 2>&1
    eval $(mantik init)
    unset MANTIK_USERNAME
    unset MANTIK_PASSWORD
    echo "MLFLOW_TRACKING_TOKEN=${MLFLOW_TRACKING_TOKEN}";
fi


srun apptainer run \
  -B ${REPO_DIR}:/opt/vissl \
  --nv \
  /p/project/deepacf/maelstrom/emmerich1/vissl.sif \
  python /opt/vissl/tools/run_distributed_engines.py \
  config=remote ${CPU_CONFIG_ARGS} \
  config.TRACK_TO_MANTIK=${TRACK_TO_MANTIK} \
  config.DISTRIBUTED.NUM_NODES=${SLURM_NNODES} \
  config.DISTRIBUTED.NUM_PROC_PER_NODE=${CPUS:-$SLURM_GPUS_PER_NODE} \
  config.DISTRIBUTED.RUN_ID=${RUN_ID} \
  config.CHECKPOINT.DIR=${CHECKPOINTS_DIR} \
  config.DATA.TRAIN.COPY_DESTINATION_DIR=${TMP_DATA_DIR} \
  config.DATA.TRAIN.DATASET_NAMES=${DATASET} \
  config.HOOKS.TENSORBOARD_SETUP.LOG_DIR=${TENSORBOARD_LOG_DIR} \
  config.LOSS.deepclusterv2_loss.num_train_samples=${N_SAMPLES} \
  config.LOSS.deepclusterv2_loss.output_dir=${CLUSTERING_DIR} \
  config.OPTIMIZER.num_epochs=${EPOCHS} \
  config.OPTIMIZER.param_schedulers.lr.lengths=${LENGTHS}
