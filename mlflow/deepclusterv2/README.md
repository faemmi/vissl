# DeepClusterV2

This MLproject uses the DeepClusterV2 (DCv2) algorithm from the
[FaceBook VISSL library](https://github.com/facebookresearch/vissl).

## General notes

* The folder structure for the configuration has to be `configs/config`.
* Each folder and subfolder within `configs/` has to contain an empty `__init__.py`.
* The model configuration is given in `configs/config/<local or remote>.yaml`.
* The dataset paths are configured in `configs/config/dataset_catalog.json`.

## Build and deploy the Apptainer image

First, set the required environment variables (see `.env.example`).

Run

```bash
make deploy
```
(or `deploy-e4`)

or, alternatively

```bash
make build
make upload
```
(or `build/upload-e4`)

## Testing locally

**Note**: To make the `dataset_catalog.json` available in the image, the `configs` directory
on the local machine needs to be bound to the respective path in the image, i.e. `-B $PWD/configs:/opt/vissl/configs`.

```bash
make install-cpu
make train
```

or

```bash
make build-apptainer
make train-apptainer
```

## Running remotely

`cd` into the git repository, then

```bash
apptainer run \
    -B $PWD:/opt/vissl/configs \
    --nv \
    <path to .sif image> \
    python /opt/vissl/tools/run_distributed_engines.py \
    config=remote
```

or submit via `sbatch`

```bash
export NODES=<number_of_nodes> GPUS=<number_of_gpus_per_node> EPOCHS=<number_of_epochs>
sbatch -A <account> --partition <partition> --nodes=${NODES} --gpus-per-node=${GPUS} --time=<time> mlflow/deepclusterv2/run.sbatch
```

Running with `sbatch` will create a folder at `/p/scratch/<account>/maelstrom/<user>/deepcluster/<SLURM job ID>`,
where the model checkpoints will be saved.

**Notes:** 

* When using the different sized datasets, one can use the following resources:
  * Daily samples: 1461 samples with a batch size of 64 fit into 4 nodes with 4 GPUs.
  * Hourly samples: 35064 samples with a batch size of 64 fit into max. 136 nodes with 4 GPUs.
  * _Remind_ that `NODES mod N_GPUS` _must be 0_, i.e. 5 nodes with 4 GPUS is not possible.
* In order to continue a failed job at the last checkpoint, set the `CHECKPOINT_FOLDER_ID` env var to the SLURM job ID of that job:
  `CHECKPOINT_FOLDER_ID=<SLURM job ID>`.
* Time limit on devel queues is `--time=02:00:00`.
* To run on CPUs only, set the `CPUS=2` env var (2 CPUs per node on JUWELS and JUWELS Booser)

  ```bash
  N_CPUS=2 sbatch ...
  ```

## Running with mantik

```bash
mantik runs submit mlflow/deepclusterv2
```

## Using JUBE for benchmarking

[JUBE](https://apps.fz-juelich.de/jsc/jube/jube2/docu/) can be used for benchmarking.
The benchmarks are defined in `jube.yaml`.
To run the benchmarks, use

```bash
jube run jube.yaml --tag jwc test
```

Replace the tags with the respective tags.
Available tags:

* test (single node/gpu, devel queues, small data sample)
* jwc (JUWELS Cluster)
* jwb (JUWELS Booster)
* e4 (E4 systems)
  * intel (Intel CPU + NVIDIA A100 GPU nodes)
  * amd (AMD CPU + AMD MI100 GPU nodes)
  * arm (ARM CPU + NVIDIA A100 GPU nodes)
    * v100 (ARM CPU + NVIDIA V100 GPU nodes)

*Note:*
For debugging consider the `--debug`, `--devel`, and/or `-v` options.

Once all runs are finnished, analysis can be performed via

```bash
jube result ap6-run/ --id <benchmark IDs> --analyse --update jube.yaml > benchmark-results.md
```

## Using AMD GPUS (ROCm)

```bash
make deploy-apptainer-rocm-e4
```

or, alternatively

```bash
make build-apptainer-rocm
make upload-apptainer-rocm-e4
```

Running with ROCm required passing
`config.OPTIMIZER.use_larc=False config.MODEL.AMP_PARAMS.USE_AMP=False`.
