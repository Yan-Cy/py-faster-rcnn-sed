import json
import os

roi_path = '/home/chenyang/sed/data/Annotations/CTE_roi'

with open('CellToEar.refine.label.json') as f:
    data = json.load(f)

for roi in data:
    image_name = str(roi[0]).replace('/','_').replace('.jpg','')
    for ix, obj in enumerate(roi[1]):
        xmin = str(int(float(obj['xmin']) / 320.0 * 720.0))
        ymin = str(int(float(obj['ymin']) / 240.0 * 576.0))
        xmax = str(int(float(obj['xmax']) / 320.0 * 720.0))
        ymax = str(int(float(obj['ymax']) / 240.0 * 576.0))
        with open(os.path.join(roi_path, image_name+'.roi'), 'a') as f:
            f.write('CellToEar ' + xmin + ' ' + ymin + ' ' + xmax + ' ' + ymax)
