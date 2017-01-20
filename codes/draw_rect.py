import Image, ImageDraw
import os
import re

"""
output_path = '/home/chenyang/Results/'
img_path = '/home/chenyang/py-faster-rcnn/data/sed/data/Images'
gt_path = '/home/chenyang/sed/data/Annotations/roi/'
result_dir = '/home/chenyang/py-faster-rcnn/data/sed/results/'
classes = ['Embrace', 'Pointing', 'CellToEar', 'Pose']
threshold = 0.0
intervals = [0.2, 0.4, 0.6, 0.8, 1.0]


def draw_dect(results, cls):
    print 'Drawing {} detection bounding box to {}'.format(len(results), cls)
    count = 0
    for result in results:
        [image_name, score, x_min, y_min, x_max, y_max] = re.split(' ', result)
        xmin = float(x_min)
        ymin = float(y_min)
        xmax = float(x_max)
        ymax = float(y_max)
        image_file = os.path.join(img_path, image_name + '.jpg')
        #print image_name
        #print image_file
        assert os.path.exists(image_file)

        for interval in intervals:
            if float(score) <= interval:
                break
        img_dir = str(interval - 0.2) + '~' + str(interval)

        # anno_file = os.path.join(output_path, cls, 'all', image_name + '.jpg')
        anno_file = os.path.join(output_path, cls, img_dir,  score + '__' + image_name + '.jpg')
        if float(score) < threshold:
            continue
        count = count + 1
        #print output_path, cls, image_name,cls,x_min,y_min,x_max,y_max
        # anno_file = os.path.join(output_path, cls, 'refine', '_'.join([image_name,cls,str(int(xmin)),str(int(ymin)),str(int(xmax)),str(int(ymax))]) + '.jpg')
        
        if os.path.exists(anno_file):
            image_file = anno_file


        image = Image.open(image_file)
        draw = ImageDraw.Draw(image)
        draw.rectangle([(xmin,ymin),(xmax,ymax)], outline='red')
        draw.text([xmin,ymin], score, fill='red')
        draw.text([xmax,ymax], cls, fill='red')

        '''Draw ground truth box'''
        gt_file = os.path.join(gt_path, image_name + '.roi')
        os.path.exists(gt_file)

        with open(gt_file) as f:
           data = f.read()
        objs = re.findall('(\S+) (\d+) (\d+) (\d+) (\d+)', data)
        for obj in objs:
            cls = obj[0]
            x1 = float(obj[1])
            y1 = float(obj[2])
            x2 = float(obj[3])
            y2 = float(obj[4])
            draw.rectangle([(x1,y1),(x2,y2)], outline='blue')
            draw.text([x1,y1], cls, fill='blue')

        ''' Get from CellToEar.json
        with open(gt_file) as f:
            gt_data = json.load(f)

        try:
            objs = next(img[1] for img in gt_data if str(img[0]).replace('/','_') == image_name + '.jpg')
        except:
            print 'Can\'t find file ' + image_file + ' in Annotation file'
            del draw
            continue

        for ix, obj in enumerate(objs):
            x1 = float(obj['xmin']) / 320 * 720
            y1 = float(obj['ymin']) / 240.0 * 576.0
            x2 = float(obj['xmax']) / 320.0 * 720.0
            y2 = float(obj['ymax']) / 240.0 * 576.0
            draw.rectangle([(x1,y1),(x2,y2)], outline='blue')
        '''

        del draw
        image.save(anno_file, 'jpeg')
    print count

file_set = '/home/chenyang/lib/ImageSets/test.txt'
all_cls = ['CellToEar', 'Embrace', 'Pointing', 'Pose']


def draw_box(file_set, gt_path, output_path, color):
    print 'Drawing ground truth file in {}, from {}, to {}, in {}'.format(file_set, gt_path, output_path, color)
    count = 0
    with open(file_set) as f:
        files = f.readlines()
    for name in files:
        name = name.strip()
        gt_file = os.path.join(gt_path, name + '.roi')
        os.path.exists(gt_file)
        img_file = os.path.join(img_path, name + '.jpg')
        
        with open(gt_file) as f:
            bboxes = f.readlines()
        for bbox in bboxes:
            bbox = bbox.strip()
            bbox = bbox.split(' ')
            
            if bbox[0] not in all_cls:
                continue
            count = count + 1
            output_file = os.path.join(output_path, bbox[0], 'refine', name + '_' + '_'.join(bbox)  + '.jpg')
        
            img = Image.open(img_file)
            draw = ImageDraw.Draw(img)
            draw.rectangle([(int(bbox[1]),int(bbox[2])),(int(bbox[3]),int(bbox[4]))], outline=color)
            del draw
            img.save(output_file, 'jpeg')
    
    print 'Total ground truth:', count
"""

def draw_bbox(src, dst, bbox, color):
    os.path.exists(src)
    img = Image.open(src)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(int(bbox[1]), int(bbox[2])), (int(bbox[3]), int(bbox[4]))], outline=color)
    img.save(dst, 'jpeg')
    del draw

def draw_rect(imgset, imgpath, rectpath, dstpath, color, all_cls):
    with open(imgset) as f:
        files = f.readlines()
    count = 0
    for name in files:
        name = name.strip()
        imgfile = os.path.join(imgpath, name + '.jpg')
        os.path.exists(imgfile)
        rectfile = os.path.join(rectpath, name + '.roi')
        os.path.exists(rectfile)

        with open(rectfile) as f:
            bboxes = f.readlines()

        for bbox in bboxes:
            bbox = bbox.strip()
            bbox = bbox.split(' ')

            if bbox[0] not in all_cls:
                continue
            count += 1
            dstfile = os.path.join(dstpath, bbox[0], name + '_' + '_'.join(bbox) + '.jpg')
            draw_bbox(imgfile, dstfile, bbox, color)

    print 'Total bbox paint:', count


if __name__ == '__main__':
    trainset = '/home/chenyang/lib/ImageSets/refine_train.txt'
    traindst = '/home/chenyang/sed_GT/train/'

    testset = '/home/chenyang/lib/ImageSets/refine_test.txt'
    testdst = '/home/chenyang/sed_GT/test/'

    imgpath = '/home/chenyang/sed/data/Images/'
    rectpath = '/home/chenyang/sed/data/Annotations/refine/'
    color = 'red'
    all_cls = ['CellToEar', 'Embrace', 'Pointing']

    draw_rect(trainset, imgpath, rectpath, traindst, color, all_cls)
    draw_rect(testset, imgpath, rectpath, testdst, color, all_cls)

    # draw results from result_dir
    #for cls in classes:
    #    result_file = os.path.join(result_dir, 'zf_'+ cls + '.txt')
    #    os.path.exists(result_file)
    #    with open(result_file) as f:
    #        results = [x.strip() for x in f.readlines()]
    #    draw_dect(results, cls)

    # draw ground truth in test.txt
    # draw_box(file_set, gt_path, output_path, 'red')
    #assert os.path.exists(result_file)
    #with open(result_file) as f:
    #    results = [x.strip() for x in f.readlines()]

    #import json
    #gt_file = '/home/chenyang/py-faster-rcnn/data/sed/data/Annotations/CellToEar.refine.label.json'
    #assert os.path.exists(gt_file)

