from sed_eval import sed_eval

if __name__ == '__main__':
    detpath = '/home/chenyang/py-faster-rcnn/data/sed/results/comp4_ffd022f0-faab-4d58-8fd0-19d881c5e028_det_test_{}.txt'
    annopath = '/home/chenyang/sed/data/Annotations'
    imagesetfile = '/home/chenyang/lib/ImageSets/test.txt'
    cachedir = None
    classes = ['CellToEar', 'Embrace']
    for cls in classes:
        print cls, ':'
        print sed_eval(detpath, annopath, imagesetfile, cls, cachedir)
