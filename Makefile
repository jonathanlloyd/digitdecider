.PHONY: train validate serve

train:
	PYTHONPATH=$(shell pwd) uv run python bin/train

validate:
	PYTHONPATH=$(shell pwd) uv run python bin/validate

serve:
	PYTHONPATH=$(shell pwd) uv run python bin/serve
