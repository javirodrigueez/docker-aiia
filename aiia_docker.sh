#!/bin/bash

docker run --gpus '"device=1"' --rm -it \
	--volume="$PWD/dataset:/workspace/dataset:ro" \
        --name pytorch_jrodriguez \
        --shm-size=16gb \
	--memory=16gb \
        jrodriguez/aiia_test bash
