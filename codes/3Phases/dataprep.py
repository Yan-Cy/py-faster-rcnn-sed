from init import cfg
from draw_rect import draw_bbox
from tools import overlap
import os
import shutil
import cv2

def cleardir(dst, subDirs = None):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.mkdir(dst)

    if subDirs != None:
        for subDir in subDirs:
            os.mkdir(os.path.join(dst, subDir))

def extract_frame(videopath, videoname, nframe, dstpath):
    #print videopath, videoname, nframe, dstpath
    videotime = int(videoname.split('_')[1])
    if videotime < 20071113:
        subdir = 'Dev08-1'
    else:
        subdir = 'Eev08-1'
    #print subdir

    cap = cv2.VideoCapture(os.path.join(videopath, subdir, videoname + '.avi'))
    cap.set(1, nframe)

    ret, frame = cap.read()

    imagename = os.path.join(dstpath, videoname + '_' + str(nframe) + '.jpg')
    ret = cv2.imwrite(imagename, frame)

    if not ret:
        return ret

    cap.release()
    return imagename


def dataprep(imgset, gtpath, videopath, rawpath, dst, cls, rootpath):
    cleardir(dst)
    for subdir in cls:
        cleardir(os.path.join(dst, subdir), ['begin', 'climax', 'end'])

    with open(imgset) as f:
        files = f.readlines()
    count = 0
    for name in files:
        name = name.strip()
        rectfile = os.path.join(gtpath, name + '.roi')
        os.path.exists(rectfile)

        videoname = '_'.join(name.split('_')[:-1])
        nframe = int(name.split('_')[-1])
        with open(os.path.join(rawpath, videoname + '.txt')) as f:
            rawdata = [x.strip().split(' ') for x in f.readlines()]

        with open(rectfile) as f:
            bboxes = f.readlines()

        for bbox in bboxes:
            bbox = bbox.strip().split(' ')

            if bbox[0] not in cls:
                continue

            oriCount = count

            for raw in rawdata:
                if len(raw) != 3:
                    continue
                beginFrame = int(raw[1])
                endFrame = int(raw[2])
                if (raw[0] == bbox[0]) and (beginFrame <= nframe) and (endFrame >= nframe):
                    midFrame = (beginFrame + endFrame) / 2

                    count += 1

                    # srcfile = os.path.join(imgpath, videoname, raw[2] + '.jpg')
                    srcfile = extract_frame(videopath, videoname, endFrame, os.path.join(rootpath, 'Images'))
                    if not srcfile:
                        continue
                    dstfile = os.path.join(dst, bbox[0], 'end',
                                           videoname + '_' + raw[2] + '_' + '_'.join(bbox) + '.jpg')
                    draw_bbox(srcfile, dstfile, bbox, 'red')
                    with open(os.path.join(dst, 'index.txt'), 'a') as f:
                        f.write(videoname + '_' + raw[2] + '\n')


                    srcfile = extract_frame(videopath, videoname, beginFrame, os.path.join(rootpath, 'Images'))
                    dstfile = os.path.join(dst, bbox[0], 'begin', videoname + '_' + raw[1] + '_' + '_'.join(bbox) + '.jpg')
                    draw_bbox(srcfile, dstfile, bbox, 'red')
                    with open(os.path.join(dst, 'index.txt'), 'a') as f:
                        f.write(videoname + '_' + raw[1] + '\n')


                    #srcfile = os.path.join(imgpath, videoname, str(midFrame) + '.jpg')
                    srcfile = extract_frame(videopath, videoname, midFrame, os.path.join(rootpath, 'Images'))
                    dstfile = os.path.join(dst, bbox[0], 'climax', videoname + '_' + str(midFrame) + '_' + '_'.join(bbox) + '.jpg')
                    draw_bbox(srcfile, dstfile, bbox, 'red')
                    with open(os.path.join(dst, 'index.txt'), 'a') as f:
                        f.write(videoname + '_' + str(midFrame) + '\n')

            if oriCount == count:
                print 'No Match Find for', str(bbox), name


    print 'Total Ground Truth:', count


def preptrain(rootpath, all_cls, phases, splitdate):
    imgsets = ['train', 'test']
    all_img = []

    for imgset in imgsets:
        for cls in all_cls:
            for phase in phases:
                imgs = os.listdir(os.path.join(rootpath, imgset, cls, phase))
                for img in imgs:
                    data = os.path.splitext(img)[0].split('_')
                    if data[0] != 'LGW':
                        continue
                    imgname = '_'.join(data[:5])
                    date = int(data[1])
                    bbox = [phase + data[5]] + [int(x) for x in data[6:]]

                    if imgname not in all_img:
                        all_img.append(imgname)
                        if date > splitdate:
                            with open(os.path.join(rootpath, 'test.txt'), 'a') as f:
                                f.write(imgname + '\n')
                        else:
                            with open(os.path.join(rootpath, 'train.txt'), 'a') as f:
                                f.write(imgname + '\n')

                    annofile = os.path.join(rootpath, 'Annotations', imgname + '.roi')
                    if not overlap(bbox, annofile):
                        with open(annofile, 'a') as f:
                            f.write(' '.join([str(x) for x in bbox]) + '\n')




if __name__ == "__main__":
    #dataprep(cfg.trainset, cfg.rectpath, cfg.videosrc, cfg.rawpath, cfg.traindst, cfg.all_cls, cfg.rootpath)
    #dataprep(cfg.testset, cfg.rectpath, cfg.videosrc, cfg.rawpath, cfg.testdst, cfg.all_cls, cfg.rootpath)

    preptrain(cfg.rootpath, ['CellToEar', 'Embrace'], cfg.phases, 20071201)