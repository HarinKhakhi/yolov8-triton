import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os import makedirs, listdir, path, remove
import time
import argparse
import cv2

from csv import DictWriter
from tqdm import tqdm

import tritonclient.grpc as grpcclient

import utils

# ================================== Parameters ==================================
TRITON_URL = 'localhost:8001'
DEFAULT_MODEL_NAME = 'yolov8_ensemble'
FILTERING_V1_MODEL_NAME = 'yolov8_ensemble_filtering_1'
MODELS = [DEFAULT_MODEL_NAME, FILTERING_V1_MODEL_NAME]

LOG_FILE = 'output/logs/gun_video_cpu.csv'
LOG_FIELDS = ['timestamp', 'input_path', 'model_name', 'total_time', 'num_boxes']
LOG_MODE = 'append'
PLOT_FILE = 'output/plots/gun_video_cpu.png'

# ================================================================================

def run_triton_model(model_name: str, input_image: np.ndarray, triton_client: grpcclient.InferenceServerClient):
    # setup input / output
    inputs = []
    outputs = []

    inputs.append(grpcclient.InferInput('images', input_image.shape, "FP32"))
    inputs[0].set_data_from_numpy(input_image)

    outputs.append(grpcclient.InferRequestedOutput('detection_boxes'))
    outputs.append(grpcclient.InferRequestedOutput('detection_scores'))
    outputs.append(grpcclient.InferRequestedOutput('detection_classes'))

    # call triton_client api to run the model 
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    # cast output to required format
    detection_boxes = results.as_numpy('detection_boxes')
    detection_scores = results.as_numpy('detection_scores')
    detection_classes = results.as_numpy('detection_classes')

    return detection_boxes, detection_scores, detection_classes


def infer(image_path):
    triton_client = utils.get_triton_client(TRITON_URL)
    log_file = open(LOG_FILE, 'a+')
    log_writer = DictWriter(log_file, fieldnames=LOG_FIELDS)

    for model_name in MODELS:
        # load model config
        expected_image_shape = triton_client.get_model_metadata(model_name).inputs[0].shape[-2:]
        
        # load image
        original_image, input_image, scale = utils.read_image(image_path, expected_image_shape)
        
        # run inference and profile and write logs
        start_time = time.perf_counter()
        detection_boxes, detection_scores, detection_classes = run_triton_model(model_name, input_image, triton_client)
        end_time = time.perf_counter()
        log_writer.writerow({
            'timestamp': round(time.time()*1000),
            'input_path': image_path,
            'model_name': model_name,
            'total_time': end_time-start_time,
            'num_boxes': len(detection_boxes)
        })

        # draw bounding boxes
        for index in range(len(detection_boxes)):
            box = detection_boxes[index]

            utils.draw_bounding_box(
                original_image, detection_classes[index], detection_scores[index],
                round(box[0] * scale), round(box[1] * scale),
                round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)
            )
        # write image
        cv2.imwrite(f'output/images/{model_name}_output.jpg', original_image)

    # de-setup
    log_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--max_images', type=int, default=1000)
    args = parser.parse_args()

    # setup
    makedirs('./output/images', exist_ok=True)
    makedirs('./output/logs', exist_ok=True)
    makedirs('./output/plots', exist_ok=True)
    if path.isfile(LOG_FILE) and LOG_MODE == 'rewrite': remove(LOG_FILE)

    # handle image
    if args.image_path:
        infer(args.image_path)
    # handle video 
    else:
        # convert video to bunch of images
        video_path = args.video_path
        video_images_path = video_path.split('.mp4')[0]

        if not path.isdir(video_images_path):
            makedirs(video_images_path)
            utils.video_to_image(video_path, video_images_path)

        # handle each image separately
        images = listdir(video_images_path)[:args.max_images]
        for filename in tqdm(images):
            infer(path.join(video_images_path, filename))
    
    # process logs
    utils.process_logs(LOG_FILE, LOG_FIELDS)
