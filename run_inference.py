import argparse
import os
import os.path as osp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tf_utils import visualization_utils as viz_utils

# Using TFv2
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, help='Saved TF model path', default=None, required=False)
parser.add_argument('--img_path', type=str, help='Path to evaluation images', default=None, required=False)
args = parser.parse_args()


class GateDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        # Load saved model and build the detection function
        self.model = tf.saved_model.load(self.model_path)

    def run_detector(self, input_img_np, min_score_thres=0.5):
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
        # boxes = detections['detection_boxes']
        # scores = detections['detection_scores']

        image_np_with_detections = input_img_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            {},
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=min_score_thres,
            agnostic_mode=False,
            skip_labels=True)

        return image_np_with_detections


def main():
    # set model and image path
    default_path = osp.dirname(osp.abspath(__file__))
    img_path = args.img_path
    model_path = args.model_path
    if img_path is None:
        img_path = osp.join(default_path, 'eval_imgs')
    if model_path is None:
        model_path = osp.join(default_path, 'tf_model/saved_model')

    detector = GateDetector(model_path)
    detector.load_model()

    # read every image in the image dir
    img_files = os.listdir(img_path)
    for f in img_files:
        image_np = np.array(Image.open(osp.join(img_path, f)))

        # run detection, default confidence threshold=0.5
        result_img = detector.run_detector(image_np)

        # show result image
        plt.figure()
        plt.imshow(result_img)
        plt.show()


if __name__ == '__main__':
    main()
