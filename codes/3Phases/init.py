from easydict import EasyDict as edict


__D = edict()
cfg = __D

__D.trainset = '/home/chenyang/lib/ImageSets/refine_train.txt'
__D.traindst = '/home/chenyang/sed_GT/train/'

__D.testset = '/home/chenyang/lib/ImageSets/refine_test.txt'
__D.testdst = '/home/chenyang/sed_GT/test/'

__D.imgpath = '/home/chenyang/sed/data/Images/'
__D.rectpath = '/home/chenyang/sed/data/Annotations/refine/'
__D.all_cls = ['CellToEar', 'Embrace', 'Pointing']

__D.rawpath = '/home/chenyang/sed/raw/txt/'
__D.phases = ['begin', 'climax', 'end']
__D.imgsrc = '/mnt/backup/chenyang/images/'
__D.rootpath = '/home/chenyang/sed_GT/'
__D.videosrc = '/mnt/backup/chenyang/video/'
