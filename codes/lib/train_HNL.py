#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from utils.cython_bbox import bbox_overlaps
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import argparse
import pprint
import numpy as np
import sys
import os
import cv2

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='voc_2007_trainval', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--testprototxt', dest='testprototxt',
                        help='test prototxt',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def combined_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        imdb = datasets.imdb.imdb(imdb_names)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

def hardNeg_learning(score, box, gt_boxes, roidb):
    if score < 0.7:
        return False
    overlaps = bbox_overlaps(
        np.ascontiguousarray([box], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    max_overlaps = overlaps.max(axis=1)
    
    if max_overlaps[0] >= cfg.TRAIN.FG_THRESH:
        return False
    roidb['boxes'].append(box)
    roidb['gt_classes'].append(0)
    roidb['gt_overlaps'].append(overlaps[0])
    roidb['seg_areas'].append(box[2] - box[0] + 1) * (box[3] - box[1] + 1) 
    return True 

CLASSES = ['__background__', 'Embrace', 'Pointing', 'CellToEar']
def get_allboxes(scores, boxes):
    num_box = len(boxes) * (len(CLASSES) - 1)
    all_score = np.zeros((num_box), dtype=np.float)
    all_box = np.zeros((num_box, 4), dtype=np.float)
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        all_box[cls_ind * len(boxes) : (cls_ind+1) * len(boxes), :] = dets[:, :4]
        all_score[cls_ind * len(boxes) : (cls_ind+1) * len(boxes)] = dets[:, -1:]

    return all_score, all_box


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    imdb, roidb = combined_roidb(args.imdb_name)
    print '{:d} roidb entries'.format(len(roidb))

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    #caffemodel = train_net(args.solver, roidb, output_dir,
    #        pretrained_model=args.pretrained_model,
    #        max_iters=args.max_iters)

    caffemodel = ['/home/chenyang/py-faster-rcnn/output/faster_rcnn_end2end/train/zf_faster_rcnn_iter_100.caffemodel']
    # Do hard negative learning
    
    imgs = [os.path.join(imdb._data_path, 'Images', x + '.jpg') for x in imdb._image_set]
    
    iters = 1
    hard_negs = []
    threhold = 100
    while True:
        print iters, 'time training end.'
        
        net = caffe.Net(args.testprototxt, caffemodel[-1], caffe.TEST)
        print '\n\nLoaded network {:s}'.format(caffemodel[-1])
        
        total_hardNeg = 0
        for im_ind, im_file in enumerate(imgs):
            print im_file
            im = cv2.imread(im_file)
            scores, boxes = get_allboxes(im_detect(net, im))

            if im_ind % 100 == 0:
                print 'Get Hard Negatives: {}/{}'.format(im_ind, len(imdb))

            gt_boxes = [roidb[im_ind]['boxes'], roidb[im_ind]['gt_classes']]
            gt_ind = np.where(gt_boxes[-1] != 0)
            gt_boxes = gt_boxes[gt_ind]

            for box_ind, box in boxes:
                if hardNeg_learning(score[box_ind], box, gt_boxes, roidb[im_ind]):
                    total_hardNeg = total_harNeg + 1
        
        hard_negs.append(total_hardNeg)
        print 'Total Hard Negative', total_hardNeg
        if total_hardNeg < threshold:
            print 'Done'
            break
        print 'Train recursively...'
        
        caffemodel = train_net(args.solver, roidb, output_dir,
                pretrained_model=caffemodel,
                max_iters=args.max_iters)
        iters = iters + 1

    print hard_negs

