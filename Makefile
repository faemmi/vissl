ROOT_DIR = $(PWD)
OUTPUT_DIR = $(HOME)/data/deepclusterv2

JSC_USER = $(MANTIK_UNICORE_USERNAME)
JSC_SSH = $(JSC_USER)@judac.fz-juelich.de
JSC_SSH_OPTIONS = -e "ssh -i $(JSC_SSH_PRIVATE_KEY_FILE)"

E4_SSH = $(E4_USERNAME)@$(E4_SERVER_IP)
E4_SSH_OPTIONS = -e "ssh -i $(E4_SSH_PRIVATE_KEY_FILE)"

VISSL_IMAGE_NAME = vissl

SSH_COPY_COMMAND = rsync -Pvra --progress

install:
	poetry install

install-cpu: install
	poetry run pip install -r requirements-cpu.txt

test:
	poetry run pytest tests/unit

train:
	@rm -rf $(OUTPUT_DIR)
	@mkdir -p $(OUTPUT_DIR)

	poetry run python tools/run_distributed_engines.py \
		config=local \
		config.OPTIMIZER.num_epochs=1 \
		config.VERBOSE=True \
		config.LOSS.deepclusterv2_loss.output_dir=$(OUTPUT_DIR) \
		config.TRACK_TO_MANTIK=False

train-apptainer:
	apptainer run \
	    -B $PWD/configs:/opt/vissl/configs \
	    -B $PWD/tests/test_data/deepclusterv2:/data \
	    mlflow/vissl.sif \
		python tools/run_distributed_engines.py \
		config=local \
		config.OPTIMIZER.num_epochs=1 \
		config.VERBOSE=True \
		config.LOSS.deepclusterv2_loss.output_dir=$(OUTPUT_DIR) \
		config.TRACK_TO_MANTIK=False

build-docker:
	sudo docker build -t $(VISSL_IMAGE_NAME):latest -f docker/vissl.Dockerfile .

build-docker-rocm:
	sudo docker build -t $(VISSL_IMAGE_NAME):latest-rocm -f docker/vissl-rocm.Dockerfile .

build-apptainer:
	sudo apptainer build --force apptainer/$(VISSL_IMAGE_NAME).sif apptainer/vissl.def

build-apptainer-rocm:
	sudo apptainer build --force apptainer/$(VISSL_IMAGE_NAME)-rocm.sif apptainer/vissl-rocm.def

upload:
	$(SSH_COPY_COMMAND) $(JSC_SSH_OPTIONS) \
		apptainer/$(VISSL_IMAGE_NAME).sif \
		$(JSC_SSH):$(JSC_PROJECT_DIR)/$(VISSL_IMAGE_NAME).sif

upload-e4:
	$(SSH_COPY_COMMAND) $(E4_SSH_OPTIONS) \
		apptainer/$(VISSL_IMAGE_NAME).sif \
		$(E4_SSH):$(E4_PROJECT_DIR)/$(VISSL_IMAGE_NAME).sif

upload-rocm-e4:
	$(SSH_COPY_COMMAND) $(E4_SSH_OPTIONS) \
		apptainer/$(VISSL_IMAGE_NAME)-rocm.sif \
		$(E4_SSH):$(E4_PROJECT_DIR)/$(VISSL_IMAGE_NAME)-rocm.sif

deploy: build-apptainer upload

deploy-e4: build-apptainer upload-e4

deploy-rocm-e4: build-apptainer-rocm upload-rocm-e4
