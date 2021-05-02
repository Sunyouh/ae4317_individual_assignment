import numpy as np
import PIL
import os
import os.path as osp
import sys
import glob
import tensorflow as tf
from barcode_detector import *
from barcode_reader import *

flags = tf.app.flags
flags.DEFINE_string('input_path', '', 'Path to images')
flags.DEFINE_string('ckpt_path', '', 'path to trained ckpt')
flags.DEFINE_string('img_path', '', 'path to the test images')
FLAGS = flags.FLAGS


def main(_):
    input_path = FLAGS.input_path
    ckpt_path = FLAGS.ckpt_path
    img_path = FLAGS.img_path

    config = tf.ConfigProto()

    # load a graph.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(ckpt_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph, config=config) as sess:

            for img_path in files:
                bboxes, base_name, width, height, depth, img = \
                    run_detection(img_path, detection_graph, sess)


if __name__ == '__main__':
    tf.app.run()
