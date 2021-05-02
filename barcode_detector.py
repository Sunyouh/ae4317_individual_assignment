import numpy as np
from PIL import Image, ImageDraw
import os.path as osp
import xml.etree.ElementTree as et


def run_detection(img_path, detection_graph, sess):
    image = Image.open(img_path)
    width, height = image.size
    depth = 3
    base_name = osp.basename(img_path)

    crop_w, crop_h = 1024, 600
    # stride_x, stride_y = int(crop_w/2), int(crop_h/2)
    stride_x, stride_y = int(crop_w/2), int(crop_h/2)

    bboxes = []

    print(width, height, base_name)

    # crop imgs
    for x in range(0, width-crop_w, stride_x):
        for y in range(0, height-crop_h, stride_y):
            cropped_img = image.crop((x, y, x+crop_w, y+crop_h))
            # c_w, c_h = cropped_img.size

            # box_cropped_img = ImageDraw.Draw(cropped_img)

            image_np_expanded = np.expand_dims(cropped_img, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')

            num_detections = detection_graph.get_tensor_by_name(
                'num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes).astype(np.int32)
            scores = np.squeeze(scores)

            for idx in range(len(boxes)):
                score = scores[idx]
                if score < 0.3:     # score thres?
                    continue
                ymin, xmin, ymax, xmax = boxes[idx][:]
                # print(boxes[0])
                p1 = (int(xmin * crop_w)+x, int(ymin * crop_h)+y)
                p2 = (int(xmax * crop_w)+x, int(ymax * crop_h)+y)

                bboxes.append((p1, p2))

    return bboxes, base_name, width, height, depth, image


def merge_bboxes(bboxes, box_img=None):
    merged_bboxes = []

    bbox = bboxes.pop(0)
    ((x1, y1), (x2, y2)) = bbox

    # left top
    while len(bboxes) > 0:
        len_bboxes = len(bboxes)
        for i in range(len_bboxes):
            (p1, p2) = bboxes.pop(0)
            diff_x = min(x2, p2[0]) - max(x1, p1[0])
            diff_y = min(y2, p2[1]) - max(y1, p1[1])

            # IOU
            area_box1 = (p2[0] - p1[0]) * (p2[1] - p1[1])
            area_box2 = (x2 - x1) * (y2 - y1)
            area_detected = float(diff_x * diff_y)
            # iou = area_detected / (area_box1 + area_box2 - area_detected)
            iou = max(area_detected / area_box2, area_detected / area_box1)

            if diff_x >= 0 and diff_y >= 0 and iou > 0.5:
                # (x2, y2) = (p2[0], p2[1])
                x1 = min(x1, p1[0], p2[0])
                y1 = min(y1, p1[1], p2[1])
                x2 = max(x2, p1[0], p2[0])
                y2 = max(y2, p1[1], p2[1])
                break
            else:
                bboxes.append((p1, p2))

            if i == len_bboxes-1:
                merged_bboxes.append(((x1, y1), (x2, y2)))
                bbox = bboxes.pop(0)
                ((x1, y1), (x2, y2)) = bbox

    merged_bboxes.append(((x1, y1), (x2, y2)))
    # print('merged into ', len(merged_bboxes))   # 42->31->18

    # for (p1, p2) in merged_bboxes:
    #     box_img.rectangle([p1, p2], fill=None, outline="green")

    return merged_bboxes


# This is a fixed form of VOC style annotation xml file
def write_label_file(bboxes, img_path, base_name, width, height, depth):
    annot = et.Element('annotation')

    folder = et.SubElement(annot, 'folder')
    folder.text = 'labels'

    filename = et.SubElement(annot, 'filename')
    filename.text = base_name

    path = et.SubElement(annot, 'path')
    path.text = img_path

    source = et.SubElement(annot, 'source')
    db = et.SubElement(source, 'database')
    db.text = 'Unknown'

    size = et.SubElement(annot, 'size')
    w = et.SubElement(size, 'width')
    w.text = str(width)
    h = et.SubElement(size, 'height')
    h.text = str(height)
    d = et.SubElement(size, 'depth')
    d.text = str(depth)

    segmented = et.SubElement(annot, 'segmented')
    segmented.text = '0'

    for (p1, p2) in bboxes:
        obj = et.SubElement(annot, 'object')
        name = et.SubElement(obj, 'name')
        name.text = 'barcode'
        pose = et.SubElement(obj, 'pose')
        pose.text = 'Unspecified'
        trunc = et.SubElement(obj, 'truncated')
        trunc.text = '0'
        diff = et.SubElement(obj, 'difficult')
        diff.text = '0'

        bndbox = et.SubElement(obj, 'bndbox')
        xmin = et.SubElement(bndbox, 'xmin')
        xmin.text = str(p1[0])
        ymin = et.SubElement(bndbox, 'ymin')
        ymin.text = str(p1[1])
        xmax = et.SubElement(bndbox, 'xmax')
        xmax.text = str(p2[0])
        ymax = et.SubElement(bndbox, 'ymax')
        ymax.text = str(p2[1])

    et.ElementTree(annot).write(img_path.split('.')[0] + '.xml')


def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)
