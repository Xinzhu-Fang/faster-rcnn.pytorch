# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
from Cindy_utils import *
import pandas as pd
import shutil
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, 1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)

    args = parser.parse_args()
    return args


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)





if __name__ == '__main__':

    args = parse_args()

    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        pascal_classes = np.asarray(['__background__',
                                     'aeroplane', 'bicycle', 'bird', 'boat',
                                     'bottle', 'bus', 'car', 'cat', 'chair',
                                     'cow', 'diningtable', 'dog', 'horse',
                                     'motorbike', 'person', 'pottedplant',
                                     'sheep', 'sofa', 'train', 'tvmonitor'])

    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        print("shitanchor")
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        with open('data/vg/objects_vocab.txt', 'r') as f:
            data = f.readlines()
        pascal_classes = np.asarray(['__background__'])
        pascal_classes = np.append(pascal_classes, np.asarray(data))
        pascal_classes = [x.strip('\n') for x in pascal_classes]

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    cfg.CUDA = True

    # import pdb; pdb.set_trace()
    # print("shitdemo")
    # print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir,
                             'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    # pdb.set_trace()

    print("load checkpoint %s" % (load_name))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable changed by Cindy
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    start = time.time()
    max_per_image = 100
    thresh = 0.05
    vis = True

    webcam_num = args.webcam_num
    # Set up webcam or get image directories
    if webcam_num >= 0:
        cap = cv2.VideoCapture(webcam_num)
        num_images = 0
    else:
        args.image_sub_dir = 'gratings'  # 'circles'  #   
        dfImages_file = os.path.join(args.image_dir, 'df_' + args.image_sub_dir + '.csv')
        dfImages = pd.read_csv(dfImages_file)
        dfImages_result = dfImages
        num_images = dfImages.shape[0]
        num_module = 7  # popout_rois.shape[0]

        # cfg.pooling

    print('Loaded Photo: {} images.'.format(num_images))
    for iPooling_size in 2**np.arange(3, 5): #8):
        iou_result = np.empty((num_images, num_module))

        for iImage in range(num_images):
            dfImage = dfImages[iImage: iImage + 1]
            total_tic = time.time()
            im_file = os.path.join(args.image_dir, args.image_sub_dir, dfImage.iloc[0]['image_file_name'])
            im_in = np.array(imread(im_file))
            if len(im_in.shape) == 2:
                im_in = im_in[:, :, np.newaxis]
                im_in = np.concatenate((im_in, im_in, im_in), axis=2)
            # rgb -> bgr
            im = im_in[:, :, ::-1]

            blobs, im_scales = _get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.data.resize_(1, 1, 5).zero_()
            num_boxes.data.resize_(1).zero_()

            det_tic = time.time()

            #popout_rois = (fasterRCNN(im_data, im_info, gt_boxes, num_boxes, iPooling_size)).cpu().numpy()
            popout_rois = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, iPooling_size)
            #pdb.set_trace()
            gt_box = np.float32(np.array([dfImage.iloc[0]['target_x'] - dfImage.iloc[0]['item_radius'], dfImage.iloc[0]['target_y'] - dfImage.iloc[0]['item_radius'], dfImage.iloc[0]['target_x'] + dfImage.iloc[0]['item_radius'], dfImage.iloc[0]['target_y'] + dfImage.iloc[0]['item_radius']]))

            #gt_box = np.float32(np.squeeze(gt_box))
            gt_box = np.repeat(gt_box[np.newaxis, :], num_module, axis=0)
            #pdb.set_trace()
            iou_result[iImage, :] = bb_iou_multi(popout_rois, gt_box)[:, 0]
            print(iImage)

            image = Image.open(im_file)
            if 'target_color_r' in dfImage.columns:
                # pdb.set_trace()

                bb_color = (int(dfImage.iloc[0]['target_color_r']), int(dfImage.iloc[0]['target_color_g']),
                            int(dfImage.iloc[0]['target_color_b']))
            else:
                bb_color = (255, 255, 255)

            for iM in range(num_module):
                image_result_dir = args.image_sub_dir + '_result_poolsize' + str(iPooling_size) + '_module' + str(iM)
                # at the dir level, image is under module, but in code, module is under image. So check module
                # for the first image
                if iImage == 0:
                    if os.path.isdir(os.path.join(args.image_dir, image_result_dir)):
                        shutil.rmtree(os.path.join(args.image_dir, image_result_dir))
                    os.mkdir(os.path.join(args.image_dir, image_result_dir))
                image0 = copy(image)    
                draw = ImageDraw.Draw(image0)
                draw.rectangle(popout_rois[iM, :], outline=bb_color)
                im_result_file = os.path.join(args.image_dir, image_result_dir, dfImage.iloc[0]['image_file_name'])
                image.save(im_result_file)
        for iM in range(num_module):
            dfImages_result['iou_poolsize' + str(iPooling_size) + '_module' + str(iM)] = iou_result[:, iM]

    dfImages_result_file = os.path.join(args.image_dir, 'df_' + args.image_sub_dir + '_result.csv')
    dfImages_result.to_csv(dfImages_result_file)

        # pdb.set_trace()
