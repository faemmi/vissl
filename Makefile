OUTPUT_DIR = "$(HOME)/data/deepclusterv2"

install:
	pip install -e .

install-cpu: install
	pip install -r requirements-cpu.txt

install-dev: install
	pip install -r requirements-dev.txt

test:
	pytest tests/unit

train:
	@rm -rf $(OUTPUT_DIR)
	@mkdir -p $(OUTPUT_DIR)

	python tools/run_distributed_engines.py \
		config=local \
		config.OPTIMIZER.num_epochs=1 \
		config.VERBOSE=True \
		config.LOSS.deepclusterv2_loss.output_dir=$(OUTPUT_DIR) \
		config.TRACK_TO_MANTIK=False

.PHONY: install install-cpu test train
