# Overview

This repository provides a comparison between ensemble models that implements different post-processing techniques. The main goal of the comparison is the find the fastest post-processing script. \
The default model is exported from [Ultralytics](https://github.com/ultralytics/ultralytics) repository with NMS post-processing.

For more information about Triton's Ensemble Models, see their documentation on [Architecture.md](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md) and some of their [preprocessing examples](https://github.com/triton-inference-server/python_backend/tree/main/examples/preprocessing).

# Directory Structure

```
server/
    start_server.sh
    models/
        yolov8_onnx/
            1/
                model.onnx
            config.pbtxt

        postprocess/
            1/
                model.py
            config.pbtxt

        <ensemble_model_versions>/
            1/
                <Empty Directory>
            config.pbtxt
client/
    main.py
README.md
```

# Quick Start

1. Install [Ultralytics](https://github.com/ultralytics/ultralytics) and TritonClient

```bash
pip install ultralytics==8.0.51 tritonclient[all]==2.31.0
```

2. Export a model to ONNX format:

```bash
yolo export model=yolov8n.pt format=onnx dynamic=True opset=16
```

3. Rename the model file to `model.onnx` and place it under the `server/models/yolov8_onnx/1` directory (see directory structure above).

4. Move to `server` directory and build the Docker Container for Triton Inference:

```bash
DOCKER_NAME="yolov8-triton"
docker build -t $DOCKER_NAME .
```

5. Run Triton Inference Server using `start_server.sh`:

```bash
./start_server.sh
```

6. Move to `client` directory and run `main.py` file and you will see the graph generated comparing the different models
