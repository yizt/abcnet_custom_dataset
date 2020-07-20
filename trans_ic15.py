# -*- coding: utf-8 -*-
"""
 @File    : trans_ic15.py
 @Time    : 2020/7/20 上午9:51
 @Author  : yizuotian
 @Description    :  转换icdar15数据集
"""
import codecs
import glob
import os

import numpy as np
from scipy import interpolate


def load_ic15(annotation_path, image_dir):
    """
    加载标注信息
    :param annotation_path:
    :param image_dir:
    :return:
    """
    image_annotation = {}
    # 文件名称，路径
    base_name = os.path.basename(annotation_path)
    image_name = base_name[3:-3] + '*'  # 通配符 gt_img_3.txt,img_3.jpg or png
    image_annotation["annotation_path"] = annotation_path
    image_annotation["image_path"] = glob.glob(os.path.join(image_dir, image_name))[0]
    image_annotation["file_name"] = os.path.basename(image_annotation["image_path"])  # 图像文件名
    # 读取边框标注
    bbox = []
    quadrilateral = []  # 四边形
    text_list = []  # gt text

    with open(annotation_path, "r", encoding='utf-8') as f:
        lines = f.read().encode('utf-8').decode('utf-8-sig').splitlines()
        # lines = f.readlines()
        # print(lines)
    for line in lines:
        line = line.strip().split(",")
        # 左上、右上、右下、左下 四个坐标 如：377,117,463,117,465,130,378,130
        lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y = map(float, line[:8])
        text = line[8]
        x_min, y_min, x_max, y_max = min(lt_x, lb_x), min(lt_y, rt_y), max(rt_x, rb_x), max(lb_y, rb_y)
        bbox.append([y_min, x_min, y_max, x_max])
        quadrilateral.append([lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y])
        text_list.append(text)

    image_annotation["boxes"] = np.asarray(bbox, np.float32).reshape((-1, 4))
    image_annotation["quadrilaterals"] = np.asarray(quadrilateral, np.float32).reshape((-1, 8))
    image_annotation["labels"] = text_list
    return image_annotation


def interp(x1, y1, x2, y2):
    """
    在(x1,y1)和(x2,y2)上均匀插值两个点
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :return:
    """
    xs = np.linspace(x1, x2, 4)
    f = interpolate.interp1d((x1, x2), (y1, y2))
    ys = f(xs)
    return xs, ys


def save_to_abc(txt_path, image_annotation):
    """
    将ic15标注数据保存为ABCNet的标注格式
    :param txt_path: 保存路径
    :param image_annotation：
    :return:
    """
    quads = image_annotation['quadrilaterals']
    labels = image_annotation['labels']
    with codecs.open(txt_path, mode='w', encoding='utf-8') as w:
        for (lt_x, lt_y, rt_x, rt_y, rb_x, rb_y, lb_x, lb_y), text in zip(quads, labels):
            xs_top, ys_top = interp(lt_x, lt_y, rt_x, rt_y)
            xs_bottom, ys_bottom = interp(rb_x, rb_y, lb_x, lb_y)
            xs = np.concatenate([xs_top, xs_bottom])
            ys = np.concatenate([ys_top, ys_bottom])
            points = np.array([xs, ys]).T  # [8,(x,y)]
            points = points.flatten()  # 打平
            # 保留两位小数
            f = np.vectorize(lambda x: round(x, 2))
            points = f(points)
            # print(points.flatten().shape)
            w.write('{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}||||{}\n'.format(*points, text))


def test_load_ic15():
    anno_path = '/Users/yizuotian/dataset/IC15/ch4_training_localization_transcription_gt/gt_img_1.txt'
    img_dir = '/Users/yizuotian/dataset/IC15/ch4_training_images'
    txt_path = './001.txt'

    save_to_abc(txt_path, load_ic15(anno_path, img_dir))


if __name__ == '__main__':
    test_load_ic15()
