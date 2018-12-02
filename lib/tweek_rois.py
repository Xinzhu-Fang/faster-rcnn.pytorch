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


def np_vec_no_jit_iou(boxes1, boxes2):
#     https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
    def run(bboxes1, bboxes2):
        x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
        x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
        xA = np.maximum(x11, np.transpose(x21))
        yA = np.maximum(y11, np.transpose(y21))
        xB = np.minimum(x12, np.transpose(x22))
        yB = np.minimum(y12, np.transpose(y22))
        interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
        boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
        boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
        iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
        return iou
    a = run(boxes1, boxes2)
    return a


def tweek_rois(my_rois):
    # rois is a tensor of shape [1,300,5]
    single_item_radius = 80
    my_rois = my_rois[0,:,:]
    num_box = my_rois.shape[0] #300
    # if float32 is specified, get error: coordinate list must contain exactly 2 coordinates
    small_rois = np.ndarray((1,4), dtype="float32")
    for iB in range(num_box-1):
        if my_rois[iB, 3] - my_rois[iB, 1] < single_item_radius and my_rois[iB, 3] - my_rois[iB, 1] < 80:
            small_rois = np.vstack((small_rois, my_rois[iB, 1:5]))
    small_rois = small_rois[1:,:]
    small_rois_centers = (small_rois[:, 0:2] + small_rois[:, 2:4])/2
    b = np_vec_no_jit_iou(small_rois, small_rois)
    c = b.sum(axis = 0)
    d = np.where(c < 2)[0] 
    # d = np.where(c < np.mean(c) - 2 * np.std(c))[0] # use mean and std

    # print(b.shape)
    # print(c.shape)
    # print(d.shape)

    print(np.mean(c))
    print(np.std(c))

    # import matplotlib.pyplot as plt
    # plt.imshow(b);
    # plt.colorbar()
    # plt.show()   
    
    clustering = MeanShift(bandwidth=single_item_radius).fit(small_rois_centers)
    num_clusters = len(set(clustering.labels_))
    mean_clusterred_rois = np.zeros((num_clusters, 4), dtype = "float32")
    for iC in range(num_clusters):
        mean_clusterred_rois[iC] = np.mean(small_rois[np.where(clustering.labels_ == iC)[0], :], axis=0)
    mean_clusterred_rois = torch.from_numpy(np.expand_dims(np.concatenate((np.zeros((1, num_clusters)).T, mean_clusterred_rois), axis=1), axis=0))
    import pdb; pdb.set_trace()
    print("shittweek")    
    return mean_clusterred_rois