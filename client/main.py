import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import sys
import argparse

TRITON_URL = 'localhost:8001'
DEFAULT_MODEL_NAME = 'yolov8_ensemble'
FILTERING_V1_MODEL_NAME = 'yolov8_ensemble_filtering_1'
MODELS = [DEFAULT_MODEL_NAME, FILTERING_V1_MODEL_NAME]

def get_triton_client(url: str = 'localhost:8001'):
    try:
        keepalive_options = grpcclient.KeepAliveOptions(
            keepalive_time_ms=2**31 - 1,
            keepalive_timeout_ms=20000,
            keepalive_permit_without_calls=False,
            http2_max_pings_without_data=2
        )
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=False,
            keepalive_options=keepalive_options
        )
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()
    return triton_client


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, color = (255, 0, )):
    label = f'({class_id}: {confidence:.2f})'
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def read_image(image_path: str, expected_image_shape) -> np.ndarray:
    expected_width, expected_height = expected_image_shape
    expected_length = min((expected_height, expected_width))

    original_image: np.ndarray = cv2.imread(image_path)
    [height, width, _] = original_image.shape
    length = max((height, width))

    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / expected_length

    input_image = cv2.resize(image, (expected_width, expected_height))
    input_image = (input_image / 255.0).astype(np.float32)

    # channel first
    input_image = input_image.transpose(2, 0, 1)

    # expand dimensions
    input_image = np.expand_dims(input_image, axis=0)
    return original_image, input_image, scale


def run_inference(model_name: str, input_image: np.ndarray, triton_client: grpcclient.InferenceServerClient):
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


def main(image_path):
    triton_client = get_triton_client(TRITON_URL)

    for model_name in MODELS:
        # load model config
        expected_image_shape = triton_client.get_model_metadata(model_name).inputs[0].shape[-2:]
        # load image
        original_image, input_image, scale = read_image(image_path, expected_image_shape)
        # run inference
        detection_boxes, detection_scores, detection_classes = run_inference(model_name, input_image, triton_client)
        # draw bounding boxes
        for index in range(len(detection_boxes)):
            box = detection_boxes[index]

            draw_bounding_box(
                original_image, detection_classes[index], detection_scores[index],
                round(box[0] * scale), round(box[1] * scale),
                round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale)
            )
        # write image
        cv2.imwrite(f'outputs/{model_name}_output.jpg', original_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./assets/bus.jpg')
    args = parser.parse_args()
    main(args.image_path)
