import shutil
import os
import random
import cv2

def cleardir(dirpath):
    if os.path.isdir(dirpath):
        os.system('rm -r ' + dirpath)
    os.system('mkdir ' + dirpath)

def calarea(bbox):
    #print bbox
    x = [float(x) for x in bbox]
    return (x[3] - x[1]) * (x[2] - x[0])

def select_data(src, imgdir, dst, classes, num, splitsize):
    cleardir(dst)

    imglist = []

    for cls in classes:
        clssrc = os.path.join(src, cls, 'climax')
        allsrc = os.listdir(clssrc)
        clsdir = os.path.join(dst, cls)
        cleardir(clsdir)
        os.mkdir(os.path.join(clsdir, 'big'))
        os.mkdir(os.path.join(clsdir, 'small'))
        
        random.shuffle(allsrc)
        total_big = 0
        total_small = 0
        for srcfile in allsrc:
            srcfile =  os.path.splitext(srcfile.strip())[0].split('_')
            #print srcfile
            if len(srcfile) != 10:
                continue
            imgname = '_'.join(srcfile[:5])
            bbox = srcfile[6:]
            if calarea(bbox) > splitsize[0]:
                if total_big > num:
                    continue
                dstimg = os.path.join(clsdir, 'big', '_'.join(srcfile[:5] + srcfile[6:]) + '.jpg')
                total_big += 1
                imglist.append(os.path.join('selected', cls, 'big', '_'.join(srcfile[:5] + srcfile[6:]) + '.jpg'))
                #print 'Big:', srcfile
            elif calarea(bbox) < splitsize[1]:
                if total_small > num:
                    continue
                dstimg = os.path.join(clsdir, 'small', '_'.join(srcfile[:5] + srcfile[6:]) + '.jpg')
                total_small += 1
                imglist.append(os.path.join('selected', cls, 'small', '_'.join(srcfile[:5] + srcfile[6:]) + '.jpg'))
                #print 'Small:', srcfile
            else:
                continue
            shutil.copy(os.path.join(imgdir, imgname + '.jpg'), dstimg)
            #im = cv2.imread(os.path.join(imgdir, imgname + '.jpg'))
            #im = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            #cv2.imwrite(dstimg, im)
        print cls, '--------------------'
        print 'Total for Big:', total_big
        print 'Total for Small:', total_small

    with open(os.path.join(dst, 'index.txt'), 'w') as f:
        f.write('\n'.join(imglist))


if __name__ == '__main__':
    src = '/home/chenyang/cydata/sed_GT/test/'
    dst = '/home/chenyang/workspace/convolutional-pose-machines-release/testing/sample_image/selected'
    imgdir = '/home/chenyang/cydata/sed_GT/Images'
    classes = ['CellToEar', 'Embrace', 'Pointing']
    num = 10
    splitsize = [20000, 7000]
    select_data(src, imgdir, dst, classes, num, splitsize)
