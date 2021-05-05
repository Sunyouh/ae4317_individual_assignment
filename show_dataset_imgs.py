import cv2
import os
import os.path as osp
import numpy as np


def read_csv(f_path, skip_header=0):
    _data = np.genfromtxt(fname=f_path, delimiter=",", skip_header=skip_header, dtype=None)
    return _data


def main():
    img_dir = 'WashingtonOBRace'
    img_path = osp.join(osp.dirname(osp.abspath(__file__)), img_dir)

    # filename, x_top_left, y_top_left, x_top_right, ..., x_bottom_left, y_bottom_left
    csv_path = osp.join(img_path, 'corners.csv')
    csv_data = read_csv(csv_path)

    img_filename = csv_data[0][0].decode("utf-8")
    im = cv2.imread(osp.join(img_path, img_filename))

    for _l in csv_data:
        # print(img_filename, _l[0])
        if img_filename != _l[0]:
            cv2.imshow('im', im)
            cv2.waitKey(0)
            img_filename = _l[0]
            im = cv2.imread(osp.join(img_path, img_filename.decode("utf-8")))

        img_file, x_t_l, y_t_l, x_t_r, y_t_r, x_b_r, y_b_r, x_b_l, y_b_l = _l
        pts = np.array([[x_t_l, y_t_l], [x_t_r, y_t_r], [x_b_r, y_b_r], [x_b_l, y_b_l]], np.int32)
        # pts = np.array([[y_t_l, x_t_l], [y_t_r, x_t_r], [y_b_r, x_b_r], [y_b_l, x_b_l]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(im, pts, True, (255, 0, 255), 10)

        cv2.rectangle(im, (x_t_l, y_t_l), (x_b_r, y_b_r), (0, 255, 255), 3)


if __name__ == '__main__':
    main()
