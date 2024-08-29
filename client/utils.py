import numpy as np
import pandas as pd
import cv2
import sys

import tritonclient.grpc as grpcclient

import seaborn as sns
import matplotlib.pyplot as plt

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


def video_to_image(video_path, target_dir):
  vidcap = cv2.VideoCapture(video_path)
  success,image = vidcap.read()

  count = 0
  while success:
    filename = f"{target_dir}/frame{count}.jpg"
    success = cv2.imwrite(filename, image)
    if not success: 
      print(f'{filename} not saved...')
      break
    success,image = vidcap.read()
    count += 1
  print(f'converted video to {count} frames')


def plot_graph(file, header, x, y):
  data = pd.read_csv(file, header=None, names=header)

  sns.violinplot(data=data, x=x, y=y)
  plt.show()
