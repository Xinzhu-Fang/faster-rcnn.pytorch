import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import math
import random
import numpy as np
import os
import pandas as pd
from copy import copy
from sklearn.cluster import MeanShift
import torch

# bb_iou_multi and single gives exxacctly the same result
def bb_iou_multi(boxes1, boxes2):
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)

    return iou


def bb_iou_single(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def tweak_rois(my_rois):
    # rois is a tensor of shape [1,300,5]
    single_item_width = 80 #600 # hardcoding
    num_candiate_box = 300 # hardcoding

    my_rois = my_rois[0, :, 1:]
    num_box = my_rois.shape[0]  # 300
    # if float32 is specified, get error: coordinate list must contain exactly 2 coordinates
    small_rois = np.ndarray((1,4), dtype="float32")
    for iB in range(num_candiate_box):
        if my_rois[iB, 2] - my_rois[iB, 0] < single_item_width and my_rois[iB, 3] - my_rois[iB, 1] < 80:
            small_rois = np.vstack((small_rois, my_rois[iB, :]))
    if small_rois.shape[0] > 1:
        small_rois = small_rois[1:, :]
    small_rois_centers = (small_rois[:, 0:2] + small_rois[:, 2:4]) / 2
    clustering = MeanShift(bandwidth=single_item_width).fit(small_rois_centers)
    num_clusters = len(set(clustering.labels_))
    mean_clustered_rois = np.zeros((num_clusters, 4), dtype="float32")
    for iC in range(num_clusters):
        mean_clustered_rois[iC] = np.mean(small_rois[np.where(clustering.labels_ == iC)[0], :], axis=0)
    mean_clustered_rois = torch.from_numpy(
        np.expand_dims(np.concatenate((np.zeros((1, num_clusters)).T, mean_clustered_rois), axis=1),
                       axis=0)).float().to('cuda')
    # import pdb; pdb.set_trace()
    # print("shittweak")    
    return mean_clustered_rois


def find_the_popout(X):
    num_items = X.shape[0]
    # import pdb; pdb.set_trace()
    # print("shittweak") 
    dists_matrix = (torch.sum(X ** 2, dim=1) - 2 * torch.mm(X, X.t())).t() + torch.sum(X ** 2, dim=1)
    sum_dists_from_others = torch.sum(dists_matrix, dim=0)
    _, popout_index = torch.max(sum_dists_from_others, 0)
    return popout_index