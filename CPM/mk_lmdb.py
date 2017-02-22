import os
import lmdb
import caffe
import caffe.proto.caffe_pb2 as pb2
import cv2
import numpy as np


def _encodedImgToDatum(encoded, label=0):
    datum = pb2.Datum()
    datum.encoded = True
    datum.data = encoded.tostring()
    datum.label = label
    return datum


def mk_lmdb(imgset, imgsrc, lmdb_dst, setname):
    with open(imgset) as f:
        imgdata = [x.strip().split(' ') for x in f.readlines]

    env = lmdb.open(os.path.join(lmdb_dst, setname), map_size=2**36)

    for i, data in enumerate(imgdata):
        imgname = data[0]
        cls = data[1]
        bbox = data[2:]
        cpmroi = cpm_preprocess(os.path.join(imgsrc, imgname), bbox)

        encoded = cv2.imencode('.jpg', cpmroi)[1]
        datum = _encodedImgToDatum(encoded)
        key = '{:0>10d}_0'.format(i)
        val = datum.SerializeToSTring()
        with env.begin(write=True) as txn:
            txn.put(key, val)

        print i, '/', len(imgdata)

    env.close()

if __name__ == '__main__':
    imgsrc = '/home/chenyang/sed/data/Images'
    lmdb_dst = '/home/chenyang/cydata/sed_subset/lmdb'
    trainset = '/home/chenyang/cydata/sed_subset/train.txt'
    mk_lmdb(trainset, imgsrc, lmdb_dst, 'train') 
    testset = '/home/chenyang/cydata/set_subset/test.txt'
    mk_lmdb(testset, imgsrc, lmdb_dst, 'test')
