import sys
import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tf_utils import visualization_utils as viz_utils

import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class GateDetector:
    def __init__(self, args=None):
        self.model_path = '/home/sunyou/projects/ae4317/ae4317_individual_assignment/tf_model/saved_model'
        self.model = None

    def load_model(self):
        # Load saved model and build the detection function
        self.model = tf.saved_model.load(self.model_path)

    def draw_boxes(self, detection_result):
        pass

    def run_detector(self, input_img_np):
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(input_img_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]
        # input_tensor = np.expand_dims(image_np, 0)

        # Run the the model
        detections = self.model(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        boxes = detections['detection_boxes']
        scores = detections['detection_scores']
        # print(boxes)
        # print(scores)

        image_np_with_detections = input_img_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            {},
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.50,
            agnostic_mode=False,
            skip_labels=True)

        # img_show = np.hstack((image_np_with_detections, input_img_np))
        plt.figure()
        plt.imshow(image_np_with_detections)
        plt.show()


def main():
    # TODO: argv
    PATH_TO_SAVED_MODEL = '/home/sunyou/projects/ae4317/ae4317_individual_assignment/tf_model/saved_model'
    IMG_PATH = '/home/sunyou/projects/ae4317/ae4317_individual_assignment/eval_imgs'

    detector = GateDetector()
    detector.load_model()

    img_files = os.listdir(IMG_PATH)

    for f in img_files:
        image_np = np.array(Image.open(osp.join(IMG_PATH, f)))

        detector.run_detector(image_np)


if __name__ == '__main__':
    main()
