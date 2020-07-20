# -*- coding: utf-8 -*-
"""
 @File    : trans_ic15.py
 @Time    : 2020/7/20 上午9:51
 @Author  : yizuotian
 @Description    :  转换icdar15数据集
"""
import argparse
import codecs
import glob
import json
import os
import sys

import cv2
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
        # 模糊标记不要
        if text == '###':
            continue
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


def ic15_to_abc(annotation_dir, image_dir, abc_gt_dir):
    """
    icdar15数据转为abcnet标注格式(带bezier控制点)
    :param annotation_dir:
    :param image_dir:
    :param abc_gt_dir:
    :return:
    """
    for ann_name in os.listdir(annotation_dir):
        ann_path = os.path.join(annotation_dir, ann_name)
        ann_info = load_ic15(ann_path, image_dir)

        dst_gt_path = os.path.join(abc_gt_dir,
                                   '{}.txt'.format(os.path.splitext(ann_info['file_name'])[0]))
        save_to_abc(dst_gt_path, ann_info)


def gen_abc_json(abc_gt_dir, abc_json_path, image_dir):
    """
    根据abcnet的gt标注生成coco格式的json标注
    :param abc_gt_dir:
    :param abc_json_path:
    :param image_dir:
    :return:
    """
    # Desktop Latin_embed.
    cV2 = [' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
           '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

    dataset = {
        'licenses': [],
        'info': {},
        'categories': [],
        'images': [],
        'annotations': []
    }
    with open('./classes.txt') as f:
        classes = f.read().strip().split()
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({
            'id': i,
            'name': cls,
            'supercategory': 'beverage',
            'keypoints': ['mean',
                          'xmin',
                          'x2',
                          'x3',
                          'xmax',
                          'ymin',
                          'y2',
                          'y3',
                          'ymax',
                          'cross']  # only for BDN
        })

    def get_category_id(cls):
        for category in dataset['categories']:
            if category['name'] == cls:
                return category['id']

    # 遍历abcnet txt 标注
    indexes = sorted([f.split('.')[0]
                      for f in os.listdir(abc_gt_dir)])
    print(indexes)

    j = 1  # 标注边框id号
    for index in indexes:
        # if int(index) >3: continue
        print('Processing: ' + index)
        im = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(index)))
        height, width, _ = im.shape
        dataset['images'].append({
            'coco_url': '',
            'date_captured': '',
            'file_name': index + '.jpg',
            'flickr_url': '',
            'id': int(index.split('_')[-1]),  # img_1
            'license': 0,
            'width': width,
            'height': height
        })
        anno_file = os.path.join(abc_gt_dir, '{}.txt'.format(index))

        with open(anno_file) as f:
            lines = [line for line in f.readlines() if line.strip()]
        for i, line in enumerate(lines):
            pttt = line.strip().split('||||')
        parts = pttt[0].split(',')
        ct = pttt[-1].strip()

        cls = 'text'
        segs = [float(kkpart) for kkpart in parts[:16]]

        xt = [segs[ikpart] for ikpart in range(0, len(segs), 2)]
        yt = [segs[ikpart] for ikpart in range(1, len(segs), 2)]
        xmin = min([xt[0], xt[3], xt[4], xt[7]])
        ymin = min([yt[0], yt[3], yt[4], yt[7]])
        xmax = max([xt[0], xt[3], xt[4], xt[7]])
        ymax = max([yt[0], yt[3], yt[4], yt[7]])
        width = max(0, xmax - xmin + 1)
        height = max(0, ymax - ymin + 1)
        if width == 0 or height == 0:
            continue

        max_len = 100
        recs = [len(cV2) + 1 for ir in range(max_len)]

        ct = str(ct)
        print('rec', ct)

        for ix, ict in enumerate(ct):
            if ix >= max_len: continue
            if ict in cV2:
                recs[ix] = cV2.index(ict)
            else:
                recs[ix] = len(cV2)

        dataset['annotations'].append({
            'area': width * height,
            'bbox': [xmin, ymin, width, height],
            'category_id': get_category_id(cls),
            'id': j,
            'image_id': int(index.split('_')[-1]),  # img_1
            'iscrowd': 0,
            'bezier_pts': segs,
            'rec': recs
        })
        j += 1

    # 写入json文件
    folder = os.path.dirname(abc_json_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(abc_json_path, 'w') as f:
        json.dump(dataset, f)


def test_load_ic15():
    anno_path = '/Users/yizuotian/dataset/IC15/ch4_training_localization_transcription_gt/gt_img_1.txt'
    img_dir = '/Users/yizuotian/dataset/IC15/ch4_training_images'
    txt_path = './001.txt'

    save_to_abc(txt_path, load_ic15(anno_path, img_dir))


def test_ic15_to_abc():
    ann_dir = '/Users/yizuotian/dataset/IC15/ch4_training_localization_transcription_gt'
    img_dir = '/Users/yizuotian/dataset/IC15/ch4_training_images'
    txt_dir = '/Users/yizuotian/dataset/IC15/abcnet_gt_train'
    json_path = '/Users/yizuotian/dataset/IC15/annotations/train.json'
    os.makedirs(txt_dir, exist_ok=True)
    ic15_to_abc(ann_dir, img_dir, txt_dir)

    gen_abc_json(txt_dir, json_path, img_dir)


def main(args):
    ic15_to_abc(args.ann_dir, args.image_dir, args.abc_gt_dir)
    gen_abc_json(args.abc_gt_dir, args.json_path, args.image_dir)


if __name__ == '__main__':
    # test_load_ic15()
    # test_ic15_to_abc()
    """
    Usage python trans_ic15.py --ann-dir /Users/yizuotian/dataset/IC15/ch4_training_localization_transcription_gt \
                 --image-dir /Users/yizuotian/dataset/IC15/ch4_training_images \
                 --abc-gt-dir /Users/yizuotian/dataset/IC15/abcnet_gt_train \
                 --json-path /Users/yizuotian/dataset/IC15/annotations/train.json
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--ann-dir", type=str, default=None)
    parse.add_argument("--image-dir", type=str, default=None)
    parse.add_argument("--abc-gt-dir", type=str, default=None)
    parse.add_argument("--json-path", type=str, default=None)
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
