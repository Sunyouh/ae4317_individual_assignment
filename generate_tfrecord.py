import tensorflow.compat.v1 as tf
import numpy as np
import os.path as osp
import io
import sys

from PIL import Image

# add tensorflow API path
sys.path.append("/home/sunyou/projects/tensorflow/models/research")
from object_detection.utils import dataset_util

flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('dataset_path', '', 'Path to the dataset')  # e.g. '/home/sunyou/projects/ae4317/WashingtonOBRace'
FLAGS = flags.FLAGS


def read_csv(f_path, skip_header=0):
    _data = np.genfromtxt(fname=f_path, delimiter=",", skip_header=skip_header, dtype=None)
    return _data


def create_tf_example(lines, dataset_path):
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    img_path = osp.join(dataset_path, lines[0][0].decode("utf-8"))
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    # print('img: ', img_path, ', w: ', width, ', h: ', height)

    filename = osp.basename(img_path).encode('utf8')
    image_format = b'png'

    for line in lines:
        img_file, x_t_l, y_t_l, x_t_r, y_t_r, x_b_r, y_b_r, x_b_l, y_b_l = line
        _xmin = float(x_t_l/width)
        _xmax = float(x_b_r/width)
        _ymin = float(y_t_l/height)
        _ymax = float(y_b_r/height)

        # make sure that every value is in RANGE (0<x,y<width,height) and
        # not inverted (min < max)
        xmin, xmax, ymin, ymax = _xmin, _xmax, _ymin, _ymax
        if xmin > xmax:
            xmin, xmax = _xmax, _xmin
        if ymin > ymax:
            ymin, ymax = _ymax, _ymin
        if ymin < 0:
            ymin = 0
        if xmin < 0:
            xmin = 0
        if xmax > width:
            xmax = width
        if ymax > height:
            ymax = height

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)

        # class is always 'gate'
        classes_text.append('Gate'.encode())
        classes.append(1)

    # print('xmin: ', xmins, ', ymins: ', ymins )
    # print('xmax: ', xmaxs, ', ymaxs: ', ymaxs)

    # make tf tensor
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
    # Using tfv1 for this script!
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # dataset csv file
    csv_path = osp.join(FLAGS.dataset_path, 'eval.csv')
    csv_data = read_csv(csv_path)

    # read csv line by line, convert the data into tf tensor and write it to tfrecord format
    lines = [csv_data[0]]
    img_filename = csv_data[0][0]
    for line in csv_data[1:]:
        if img_filename == line[0]:
            lines.append(line)
            continue
        else:
            tf_example = create_tf_example(lines, FLAGS.dataset_path)
            writer.write(tf_example.SerializeToString())
            img_filename = line[0]
            lines = [line]

    writer.close()


if __name__ == '__main__':
    tf.app.run()
