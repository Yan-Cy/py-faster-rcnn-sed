import shutil
import sys
import os
import random
import cv2
sys.path.append('../codes/')
from sed_toolbox import *

'''
Randomly select num[class] images (area > splitsize) from source, exclude all images in excludesets.
Images will be copied to dst/images/
Images with bounding box (in annopath) will be saved to dst/bboximg/.
Images' set will be saved to dst/ with its setname
'''
def select_sedsubset_data(srcs, imgdir, dst, classes, num, splitsize, excludesets, setname, annopath):
    #cleardir(dst)
    #cleardir(os.path.join(dst), 'images')
    #cleardir(os.path.join(dst), 'bboximg')
   
    excludelist = []
    for excludeset in excludesets:
        with open(excludeset) as f:
            imgset = [x.strip() for x in f.readlines()]
        excludelist += imgset
    

    imglist = []
    for src in srcs:
        with open(src) as f:
            imgset = [x.strip() for x in f.readlines()]
        imglist += imgset

    random.shuffle(imglist)


    bboxlist = []
    for imgname in imglist:
        annofile = os.path.join(annopath, imgname + '.roi')
        with open(annofile) as f:
            annos = [x.strip().split(' ') for x in f.readlines()]
        annos = [[imgname] + x for x in annos if len(x) == 5 and x[0] in classes]
        bboxlist += annos


    random.shuffle(bboxlist)

    selectlist = []
    print len(bboxlist)
    for imgdata in bboxlist:
        if sum(num[x] for x in num) < 1:
            break
      
        #print imgdata

        imgname = imgdata[0]
        date = int(imgname.split('_')[1])
        cls = imgdata[1]
        bbox = imgdata[2:]

        if (setname == 'train' and date > 20071113) or (setname == 'test' and date < 20071113):
            continue
        if (num[cls] < 1) or (calarea(bbox) < splitsize) or (imgname in excludelist):
            continue
      
        num[cls] -= 1
        selectlist.append(imgdata)
        srcimg = os.path.join(imgdir, imgname + '.jpg')
        dstimg = os.path.join(dst, 'images', imgname + '.jpg')
        shutil.copy(srcimg, dstimg)
        bboximg = os.path.join(dst, 'bboximg', '_'.join([imgname, cls] + bbox) + '.jpg')
        draw_bbox(srcimg, bboximg, bbox, 'red')

    setfile = os.path.join(dst, setname + '.txt')
    with open(setfile, 'w') as f:
        for imgdata in selectlist:
            f.write(' '.join(imgdata) + '\n')

    print num
    

def analyze_set(allset, classes):
    lower1113 = {}
    higher1113 = {}
    for cls in classes:
        lower1113[cls] = 0
        higher1113[cls] = 0

    for setpath in allset:
        with open(setpath) as f:
            imgset = [x.strip().split(' ') for x in f.readlines()]
        for img in imgset:
            date = int(img[0].split('_')[1])
            if date < 20071113:
                lower1113[img[1]] += 1
            else:
                higher1113[img[1]] += 1

    total_lower = 0
    total_higher = 0
    for cls in classes:
        total_lower += lower1113[cls]
        total_higher += higher1113[cls]

    print '< 1113: ', total_lower
    print lower1113
    print '> 1113: ', total_higher
    print higher1113


if __name__ == '__main__':
    sys.path.append('../codes')
    
    classes = ['CellToEar', 'Embrace', 'Pointing', 'Pose']
    allset = ['/home/chenyang/cydata/sed_subset/annodata/train.txt', '/home/chenyang/cydata/sed_subset/annodata/test.txt']
    #analyze_set(allset, classes) 
    
    src = ['/home/chenyang/lib/ImageSets/refine_train.txt', '/home/chenyang/lib/ImageSets/refine_test.txt', '/home/chenyang/lib/ImageSets/pose_train_filter.txt', '/home/chenyang/lib/ImageSets/pose_test_filter.txt']
    annopath = '/home/chenyang/sed/data/Annotations/general_roi/'
    dst = '/home/chenyang/cydata/sed_subset/appenddata/'
    imgdir = '/home/chenyang/sed/data/Images/'
    excludesets = ['/home/chenyang/cydata/sed_subset/annodata/trainset.txt', '/home/chenyang/cydata/sed_subset/annodata/testset.txt']
    num = {}
    num['CellToEar'] = 62
    num['Embrace'] = 69
    num['Pointing'] = 95
    num['Pose'] = 286
    splitsize = 6000
    select_sedsubset_data(src, imgdir, dst, classes, num, splitsize, excludesets, 'train', annopath)

    num['CellToEar'] = 88
    num['Embrace'] = 81
    num['Pointing'] = 55
    num['Pose'] = 164
 
    select_sedsubset_data(src, imgdir, dst, classes, num, splitsize, excludesets, 'test', annopath)

    

