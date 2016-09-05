import Image, ImageDraw
import os
import re

gt_path = '/home/chenyang/sed/data/Annotations/roi/'
result_dir = '/home/chenyang/py-faster-rcnn/data/sed/results/'
files = ['VGG_Embrace', 'VGG_Pointing', 'VGG_CellToEar']
intervals = [0.2, 0.4, 0.6, 0.8, 1.0]


def draw_rect(results, file):
    for result in results:
        [image_name, score, x_min, y_min, x_max, y_max] = re.split(' ', result)
        x_min = float(x_min)
        y_min = float(y_min)
        x_max = float(x_max)
        y_max = float(y_max)
        image_file = os.path.join('/home/chenyang/py-faster-rcnn/data/sed/data/Images', image_name + '.jpg')
        #print image_name
        #print image_file
        assert os.path.exists(image_file)

        for interval in intervals:
            if float(score) <= interval:
                break
        img_dir = str(interval - 0.2) + '~' + str(interval)

        anno_file = os.path.join('/home/chenyang/Results/', file, img_dir,  score + '__' + image_name + '.jpg')
        if os.path.exists(anno_file):
            image_file = anno_file


        image = Image.open(image_file)
        draw = ImageDraw.Draw(image)
        draw.rectangle([(x_min,y_min),(x_max,y_max)], outline='red')
        draw.text([(x_min+x_max)/2.0,(y_min+y_max)/2.0], score, fill='red')

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
            draw.text([x1,y1], score, fill='blue')

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
     

if __name__ == '__main__':
    for file in files:
        result_file = os.path.join(result_dir, file + '.txt')
        os.path.exists(result_file)
        with open(result_file) as f:
            results = [x.strip() for x in f.readlines()]
        draw_rect(results, file)
    #assert os.path.exists(result_file)
#with open(result_file) as f:
#    results = [x.strip() for x in f.readlines()]

#import json
#gt_file = '/home/chenyang/py-faster-rcnn/data/sed/data/Annotations/CellToEar.refine.label.json'
#assert os.path.exists(gt_file)

