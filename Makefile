.PHONY: help run build clean local

SERVICE_NAME ?= "processor_tiff_zarr_converter"

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make build   - build the Docker image"
	@echo "make run     - run the processor locally via docker-compose"
	@echo "make clean   - remove output files"

build:
	docker build -t $(SERVICE_NAME) .

run:
	docker-compose -f docker-compose.yml down --remove-orphans
	docker-compose -f docker-compose.yml build
	docker-compose -f docker-compose.yml up --exit-code-from processor

local:
	INPUT_DIR=./test_input OUTPUT_DIR=./test_output .venv/bin/python -m processor.main

clean:
	rm -rf test_output/*
