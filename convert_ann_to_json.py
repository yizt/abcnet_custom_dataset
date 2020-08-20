# -*- coding: utf-8 -*-
"""
 @File    : convert_ann_to_json.py
 @Time    : 2020-8-17 16:13
 @Author  : yizuotian
 @Description    : 生成windows_label_tool工具的标注格式转换为ABCNet训练的json格式标注
"""
import argparse
import json
import os
import sys
import cv2
from AdelaiDet.tools import bezier_utils
import numpy as np


def gen_abc_json(abc_gt_dir, abc_json_path, image_dir, classes_path):
    """
    根据abcnet的gt标注生成coco格式的json标注
    :param abc_gt_dir: windows_label_tool标注工具生成标注文件目录
    :param abc_json_path: ABCNet训练需要json标注路径
    :param image_dir:
    :param classes_path: 类别文件路径
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
    with open(classes_path) as f:
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
        # print('Processing: ' + index)
        im = cv2.imread(os.path.join(image_dir, '{}.jpg'.format(index)))
        im_height, im_width = im.shape[:2]
        dataset['images'].append({
            'coco_url': '',
            'date_captured': '',
            'file_name': index + '.jpg',
            'flickr_url': '',
            'id': int(index.split('_')[-1]),  # img_1
            'license': 0,
            'width': im_width,
            'height': im_height
        })
        anno_file = os.path.join(abc_gt_dir, '{}.txt'.format(index))

        with open(anno_file) as f:
            lines = [line for line in f.readlines() if line.strip()]
        # 没有清晰的标注，跳过
        if len(lines) <= 1:
            continue
        for i, line in enumerate(lines[1:]):
            elements = line.strip().split(',')
            polygon = np.array(elements[:28]).reshape((-1, 2)).astype(np.float32)  # [14,(x,y)]
            control_points = bezier_utils.polygon_to_bezier_pts(polygon, im)  # [8,(x,y)]
            ct = elements[-1].replace('"', '').strip()

            cls = 'text'
            # segs = [float(kkpart) for kkpart in parts[:16]]
            segs = [float(kkpart) for kkpart in control_points.flatten()]
            xt = [segs[ikpart] for ikpart in range(0, len(segs), 2)]
            yt = [segs[ikpart] for ikpart in range(1, len(segs), 2)]

            # 过滤越界边框
            if max(xt) > im_width or max(yt) > im_height:
                print('The annotation bounding box is outside of the image:{}'.format(index))
                print("max x:{},max y:{},w:{},h:{}".format(max(xt), max(yt), im_width, im_height))
                continue
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
            # print('rec', ct)

            for ix, ict in enumerate(ct):
                if ix >= max_len:
                    continue
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


def main(args):
    gen_abc_json(args.ann_dir, args.dst_json_path, args.image_dir, args.classes_path)


if __name__ == '__main__':
    """
    Usage: python convert_ann_to_json.py \
    --ann-dir /path/to/gt \
    --image-dir /path/to/image \
    --dst-json-path train.json 
    """
    parse = argparse.ArgumentParser()
    parse.add_argument("--ann-dir", type=str, default=None)
    parse.add_argument("--image-dir", type=str, default=None)
    parse.add_argument("--dst-json-path", type=str, default=None)
    parse.add_argument("--classes-path", type=str, default='./classes.txt')
    arguments = parse.parse_args(sys.argv[1:])
    main(arguments)
