import tensorflow as tf
import numpy as np
import os.path as osp
import io

from PIL import Image

from object_detection.utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('dataset_path', '/home/sunyou/projects/ae4317/ae4317_individual_assignment/WashingtonOBRace', 'Path to the dataset')
FLAGS = flags.FLAGS


def read_csv(f_path, skip_header=0):
    _data = np.genfromtxt(fname=f_path, delimiter=",", skip_header=skip_header, dtype=None)
    return _data


def create_tf_example(lines):
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    img_path = osp.join(flags.dataset_path, lines[0][0].decode("utf-8"))
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = osp.basename(img_path).encode('utf8')
    image_format = b'jpg'

    for line in lines:
        img_file, x_t_l, y_t_l, x_t_r, y_t_r, x_b_r, y_b_r, x_b_l, y_b_l = line
        xmins.append(float(x_t_l/width))
        xmaxs.append(float(x_b_r/width))
        ymins.append(float(y_t_l/height))
        ymaxs.append(float(y_b_r/height))
        classes_text = 'Gate'
        classes = 1

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    csv_path = osp.join(flags.dataset_path, 'corners.csv')
    csv_data = read_csv(csv_path)

    lines = [csv_data[0]]
    img_filename = csv_data[0][0]
    for line in csv_data[1:]:
        if img_filename == line[0]:
            lines.append(line)
            continue
        else:
            tf_example = create_tf_example(lines)
            writer.write(tf_example.SerializeToString())
            img_filename = line[0]
            lines = [line]

    writer.close()


if __name__ == '__main__':
    tf.app.run()
