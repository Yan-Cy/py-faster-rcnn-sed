import Image, ImageDraw
import os
import re

def draw_bbox(src, dst, bbox, color):
    '''
    Paint src image with its bounding box in color
    Save it to dst
    '''
    os.path.exists(src)
    img = Image.open(src)
    draw = ImageDraw.Draw(img)
    draw.rectangle([(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))], outline=color)
    img.save(dst, 'jpeg')
    del draw

def cleardir(dirpath):
    '''
    Clear the directory dirpath
    '''
    if os.path.isdir(dirpath):
        os.system('rm -r ' + dirpath)
    os.system('mkdir ' + dirpath)

def calarea(bbox):
    '''
    Calculate the area of a bounding box
    '''
    x = [float(x) for x in bbox]
    return (x[3] - x[1]) * (x[2] - x[0])


