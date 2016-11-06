import os
import shutil
import re
import collections

cams = ['LGW_20071206_E1_CAM1', 'LGW_20071206_E1_CAM2', 'LGW_20071206_E1_CAM3', 'LGW_20071206_E1_CAM4', 'LGW_20071206_E1_CAM5', 
        'LGW_20071207_E1_CAM1', 'LGW_20071207_E1_CAM2', 'LGW_20071207_E1_CAM3', 'LGW_20071207_E1_CAM4', 'LGW_20071207_E1_CAM5']
raw_data = '/mnt/sdc/chenyang/raw_data/raw_per_5/'
img_db = '/mnt/sdc/chenyang/sed/data/Images/'

def prepare_db():
    ftest = open('1206test.txt', 'w')
    for cam in cams:
        data_path = os.path.join(raw_data, cam)
        imgs = os.listdir(data_path)
        for img in imgs:
            imgname = cam + '_' + os.path.splitext(img)[0]
            src = os.path.join(data_path, img)
            dst = os.path.join(img_db, imgname + '.jpg')
            if imgname[-1] != '5':
                continue
            print imgname, src, dst
            #shutil.copy(src, dst)
            ftest.write(imgname + '\n')

CLASSES = ['Embrace', 'Pointing', 'CellToEar']
dettemplate = ''

def prepare_csv():
    detcsv = dict()
    
    for cls in CLASSES:
        detfile = dettemplate.format(cls)
        with open(detfile) as f:
            dets = [x.strip().split(' ') for x in f.readlines()]
        for det in dets:
            data = det[0].split('_')
            imgname = '_'.join(det[:-1])
            frame = int(det[:-1])
            score = float(det[1])
            x1 = det[2]
            y1 = det[3]
            x2 = det[4]
            y2 = det[5]
            
            if not detcsv.has_key(imgname):
                detcsv[imgname] = []
            detcsv[imgname].append([frame, cls, score, x1, y1, x2, y2])

    for imgname in detcsv:
        dets = sorted(detcsv[imgname])
        spans = []
        left = dets[0][0]
        cursor = left
        right = -1
        id = 1
        for det in dets:
            if id == 1:
                id = id + 1
                continue

        csvfile = os.path.join('csv', imgname + '.csv')
        with open(csvfile, 'w') as f:
            f.write('"ID","EventType","Framespan","DetectionScore","DetectionDecision"\n')
            id = 1
            for cls, score, x1


if __name__ == '__main__':
    #prepare_db()
    prepare_csv()
    xml_script()
