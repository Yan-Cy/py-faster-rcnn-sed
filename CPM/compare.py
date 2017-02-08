import os
import math

def distance(p1, p2):
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    return math.sqrt(dx * dx + dy * dy)

src_part = ['Head', 'Neck', 'Lsho', 'Rsho', 'Lelb', 'Relb', 'Lwri', 'Rwri']
src_to_dst = [0,1,5,2,6,3,7,4]

def compare(imglist, src, dst, setname):
    dis = [0.0] * len(src_to_dst)
    count = [0] * len(src_to_dst)

    with open(imglist) as f:
        imgset = [x.strip().split(' ') for x in f.readlines()]

    for annofile in imgset:
        filename = '_'.join(annofile) + '.txt'
        
        srcfile = os.path.join(src, filename)
        with open(srcfile) as f:
            srcdata = [x.strip().split(' ') for x in f.readlines()]
        if len(srcdata) == 0:
            continue

        dstfile = os.path.join(dst, filename)
        with open(dstfile) as f:
            dstdata = [x.strip().split(' ') for x in f.readlines()]

        for i in xrange(len(dis)):
            if srcdata[i][0] == '-1':
                continue
            dis[i] += distance(srcdata[i][0:2], dstdata[src_to_dst[i]][0:2])
            count[i] += 1

    print '-----------------'
    print setname, ':'
    for i in xrange(len(dis)):
        print src_part[i], dis[i] / count[i] 

if __name__ == '__main__':
    trainset = '/home/chenyang/cydata/sed_subset/annodata/train.txt'
    testset = '/home/chenyang/cydata/sed_subset/annodata/test.txt'
    trainanno = '/home/chenyang/cydata/sed_subset/annodata/train_annos'
    testanno = '/home/chenyang/cydata/sed_subset/annodata/test_annos'
    CPManno = './results'
    compare(trainset, trainanno, CPManno, 'train')
    compare(testset, testanno, CPManno, 'test')
