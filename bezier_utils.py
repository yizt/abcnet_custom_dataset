# -*- coding: utf-8 -*-
"""
 @File    : bezier_utils.py
 @Time    : 2020-8-18 14:23
 @Author  : yizuotian
 @Description    :
""" 

import math

import numpy as np
import torch
from scipy.special import comb as n_over_k
from shapely.geometry import *
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch import nn

Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
BezierCoeff = lambda ts: [[Mtk(3, t, k) for k in range(4)] for t in ts]


class Bezier(nn.Module):
    def __init__(self, ps, ctps):
        """
        ps: numpy array of points
        """
        super(Bezier, self).__init__()
        self.x1 = nn.Parameter(torch.as_tensor(ctps[0], dtype=torch.float64))
        self.x2 = nn.Parameter(torch.as_tensor(ctps[2], dtype=torch.float64))
        self.y1 = nn.Parameter(torch.as_tensor(ctps[1], dtype=torch.float64))
        self.y2 = nn.Parameter(torch.as_tensor(ctps[3], dtype=torch.float64))
        self.x0 = ps[0, 0]
        self.x3 = ps[-1, 0]
        self.y0 = ps[0, 1]
        self.y3 = ps[-1, 1]
        self.inner_ps = torch.as_tensor(ps[1:-1, :], dtype=torch.float64)
        self.t = torch.as_tensor(np.linspace(0, 1, 81))

    def forward(self):
        x0, x1, x2, x3, y0, y1, y2, y3 = self.control_points()
        t = self.t
        bezier_x = (1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * (
                (1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3))
        bezier_y = (1 - t) * ((1 - t) * ((1 - t) * y0 + t * y1) + t * ((1 - t) * y1 + t * y2)) + t * (
                (1 - t) * ((1 - t) * y1 + t * y2) + t * ((1 - t) * y2 + t * y3))
        bezier = torch.stack((bezier_x, bezier_y), dim=1)
        diffs = bezier.unsqueeze(0) - self.inner_ps.unsqueeze(1)
        sdiffs = diffs ** 2
        dists = sdiffs.sum(dim=2).sqrt()
        min_dists, min_inds = dists.min(dim=1)
        return min_dists.sum()

    def control_points(self):
        return self.x0, self.x1, self.x2, self.x3, self.y0, self.y1, self.y2, self.y3

    def control_points_f(self):
        return self.x0, self.x1.item(), self.x2.item(), self.x3, self.y0, self.y1.item(), self.y2.item(), self.y3


def train(x, y, ctps, lr):
    x, y = np.array(x), np.array(y)
    ps = np.vstack((x, y)).transpose()
    bezier = Bezier(ps, ctps)

    return bezier.control_points_f()


def bezier_fit(x, y):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dt = (dx ** 2 + dy ** 2) ** 0.5
    t = dt / dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()

    data = np.column_stack((x, y))
    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (9,4) -> (4,9)

    control_points = Pseudoinverse.dot(data)  # (4,9)*(9,2) -> (4,2)
    medi_ctp = control_points[1:-1, :].flatten().tolist()
    return medi_ctp


def bezier_fitv2(x, y):
    xc01 = (2 * x[0] + x[-1]) / 3.0
    yc01 = (2 * y[0] + y[-1]) / 3.0
    xc02 = (x[0] + 2 * x[-1]) / 3.0
    yc02 = (y[0] + 2 * y[-1]) / 3.0
    control_points = [xc01, yc01, xc02, yc02]
    return control_points


def is_close_to_line(xs, ys, thres):
    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(xs.reshape(-1, 1), ys.reshape(-1, 1))
    # Predict
    y_predicted = regression_model.predict(xs.reshape(-1, 1))

    # model evaluation
    rmse = mean_squared_error(ys.reshape(-1, 1) ** 2, y_predicted ** 2)
    rmse = rmse / (ys.reshape(-1, 1) ** 2 - y_predicted ** 2).max() ** 2

    if rmse > thres:
        return 0.0
    else:
        return 2.0


def is_close_to_linev2(xs, ys, size, thres=0.05):
    pts = []
    nor_pixel = int(size ** 0.5)
    for i in range(len(xs)):
        pts.append(Point([xs[i], ys[i]]))
    # iterate by pairs of points
    slopes = [(second.y - first.y) / (second.x - first.x) if not (second.x - first.x) == 0.0 else math.inf * np.sign(
        (second.y - first.y)) for first, second in zip(pts, pts[1:])]
    st_slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    max_dis = ((ys[-1] - ys[0]) ** 2 + (xs[-1] - xs[0]) ** 2) ** (0.5)

    diffs = abs(slopes - st_slope)
    score = diffs.sum() * max_dis / nor_pixel

    if score < thres:
        return 0.0
    else:
        return 3.0


def polygon_to_bezier_pts(polygons, img):
    """
    多边形转为贝塞尔控制点
    :param polygons: [N,(x,y)] 左上角为起始点，顺时针排列
    :param img: [H,W,C]
    :return:
    """
    assert len(polygons) % 2 == 0
    mid_idx = len(polygons) // 2

    curve_data_top = polygons[:mid_idx]
    curve_data_bottom = polygons[mid_idx:]

    x_data = curve_data_top[:, 0]
    y_data = curve_data_top[:, 1]
    # print('x_data len:{},x_data:{}'.format(len(x_data), x_data))

    init_control_points = bezier_fit(x_data, y_data)
    # print('init_control_points len:{},init_control_points{}'.format(len(init_control_points),
    #                                                                 init_control_points))

    learning_rate = is_close_to_linev2(x_data, y_data, img.size)

    x0, x1, x2, x3, y0, y1, y2, y3 = train(x_data, y_data, init_control_points, learning_rate)

    # print(x0, x1, x2, x3, y0, y1, y2, y3)

    x_data_b = curve_data_bottom[:, 0]
    y_data_b = curve_data_bottom[:, 1]

    init_control_points_b = bezier_fit(x_data_b, y_data_b)

    learning_rate = is_close_to_linev2(x_data_b, y_data_b, img.size)

    x0_b, x1_b, x2_b, x3_b, y0_b, y1_b, y2_b, y3_b = train(x_data_b, y_data_b, init_control_points_b, learning_rate)
    control_points = np.array([
        [x0, y0],
        [x1, y1],
        [x2, y2],
        [x3, y3],
        [x0_b, y0_b],
        [x1_b, y1_b],
        [x2_b, y2_b],
        [x3_b, y3_b]
    ])
    return control_points
