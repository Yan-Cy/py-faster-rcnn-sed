import os
import re

anno_path = '/home/chenyang/sed/data/Annotations/'
#roi_dbs = ['raw_bbox', 'pose_roi']
roi_dbs = ['refine', 'pose_roi']
final_path = '/home/chenyang/sed/data/Annotations/roi/'

def merge_roidb(roi_dbs):
    for roi_db in roi_dbs:
        roi_path = os.path.join(anno_path, roi_db)
        roi_files = os.listdir(roi_path)
        for roi_file in roi_files:
            f_roi = open(os.path.join(roi_path, roi_file))
            final_roi = os.path.join(final_path, roi_file)
            if os.path.exists(final_roi):
                print roi_file
            with open(final_roi, 'a') as f:
                f.write(f_roi.read())

files = ['/home/chenyang/lib/ImageSets/train.txt', '/home/chenyang/lib/ImageSets/test.txt']
# src = ['/home/chenyang/sed/data/Annotations/refine/', '/home/chenyang/sed/data/Annotations/pose_roi/'] 
src = ['/home/chenyang/sed/data/Annotations/refine/']
dst = '/home/chenyang/sed/data/Annotations/roi/'
def extract_roidb(files, src, dst):
    all_imgs = []
    for img_set in files:
        with open(img_set) as f:
            imgs = [x.strip() for x in f.readlines()]
        for img in imgs:
            if img in all_imgs:
                continue
            all_imgs.append(img)
            found = 0
            roi = os.path.join(dst, img + '.roi')
            if os.path.exists(roi):
                print img
            for roidb in src:
                src_roi = os.path.join(roidb, img + '.roi')
                if not os.path.exists(src_roi):
                    continue
                found = found + 1

                src_file = open(src_roi)
                with open(roi, 'a') as f:
                    f.write(src_file.read())

            if found == 0:
                print 'Not find roi of image {} from src'.format(img)
                with open(roi, 'w') as f:
                    f.write('')

def filter_roidb(dst):
    roidbs = os.listdir(dst)
    for roidb in roidbs:
        roi_file = os.path.join(dst, roidb)
        with open(roi_file) as f:
            rois = [list(x) for x in re.findall('(\S+) (\d+) (\d+) (\d+) (\d+)', f.read())]
        change = False
        for roi in rois:
            if int(roi[1]) > 719:
                change = True
                roi[1] = '719'
            if int(roi[2]) > 575:
                change = True
                roi[2] = '575'
            if int(roi[3]) > 719:
                change = True
                roi[3] = '719'
            if int(roi[4]) > 757: 
                change = True
                roi[4] = '575'
        if change:
            print 'refine -> ', roidb
            with open(roi_file, 'w') as f:
                for roi in rois:
                    f.write(' '.join(roi) + '\n')

if __name__ == '__main__':
    #merge_roidb(roi_dbs)
    extract_roidb(files, src, dst)
    filter_roidb(dst)
