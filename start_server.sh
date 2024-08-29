docker run \
-p8000:8000 -p8001:8001 -p8002:8002 \
-it --rm \
-v ./models:/models \
yolov8-triton