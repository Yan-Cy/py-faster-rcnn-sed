import os
import re
import numpy as np

CLASSES = ['Embrace', 'Pointing', 'CellToEar']
filename = '/home/chenyang/sed/results/comp4_97bab66b-df89-45b6-a9f4-b37812f6c82a_det_test_{}.txt'
annopath = '/home/chenyang/sed/data/Annotations/'
imagesetfile = '/home/chenyang/sed/data/ImageSets/test.txt'

miss_num = dict()

def parse_rec(annopath, imagename):
    objects = []
    imagepath = os.path.join(annopath, 'roi', imagename + '.roi')
    with open(imagepath) as f:
        data = f.read()
    objs = re.findall('(\S+) (\d+) (\d+) (\d+) (\d+)', data)
    objs = [x for x in objs if x[0] in CLASSES]
    for ix, obj in enumerate(objs):
        obj_struct = {}
        obj_struct['bbox'] = [  float(obj[1]),
                                float(obj[2]),
                                float(obj[3]),
                                float(obj[4])]
        obj_struct['name'] = obj[0]
        objects.append(obj_struct)
    return objects

ovthresh = 0.5
def get_roi_cls(roidb, gt_roidb):
    R = [obj for obj in gt_roidb]
    BBGT = np.array([x['bbox'] for x in R])
    #print BBGT.shape
    gt_cls = np.array([x['name'] for x in R])

    score = float(roidb[1])
    bb = np.array(roidb[2:], dtype=np.float)
    #print roidb
    #print bb
    ovmax = -np.inf

    if BBGT.size > 0:
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

    if ovmax > ovthresh:
        #if score < 0.7:
        #    miss_num[gt_cls[jmax]] = miss_num[gt_cls[jmax]] + 1
        return gt_cls[jmax]
    else:
        #print ovmax
        return 'background'

def draw_confusion_matrix(filename):
    matrix = dict()
    cls_num = len(CLASSES)

    with open(imagesetfile, 'r') as f:
         lines = f.readlines()
    imagenames = [x.strip() for x in lines]

    gt_num = dict()
    for cls in CLASSES:
        gt_num[cls] = 0
        miss_num[cls] = 0
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_rec(annopath, imagename)
        for cls in CLASSES:
            gt_num[cls] = gt_num[cls] + len([x for x in recs[imagename] if x['name'] == cls]) 
    print gt_num

    for cls in CLASSES:
        cls_file = filename.format(cls)
        with open(cls_file, 'r') as f:
            roidbs = f.readlines()
        roidbs = [x.strip().split(' ') for x in roidbs]

        refine_roidb = [x for x in roidbs if float(x[1]) > 0.95]
        #miss_num[cls] = len(roidbs) - len(refine_roidb)
        roidbs = refine_roidb

        for roidb in roidbs:
            roi_cls = get_roi_cls(roidb, recs[roidb[0]])
           
            if roi_cls == cls and float(roidb[1]) < 0.7:
                miss_num[cls] = miss_num[cls] + 1
                #print cls, float(roidb[1])
            if not matrix.has_key(roi_cls):
                matrix[roi_cls] = dict()
            if matrix[roi_cls].has_key(cls):
                matrix[roi_cls][cls] = matrix[roi_cls][cls] + 1
            else:
                matrix[roi_cls][cls] = 0
    
    print miss_num
    for cls in matrix.keys():
        print cls, matrix[cls]

if __name__ == '__main__':
    draw_confusion_matrix(filename)
