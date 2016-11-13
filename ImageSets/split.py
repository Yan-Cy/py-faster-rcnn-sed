import re

splitdate = 1113
infile = 'refine_train.txt'
outfile = '1113_refine.txt'

def split(infile, outfile, splitdate):
 
    with open(infile) as f:
        data = [x.strip() for x in f.readlines()]

    with open(outfile, 'w') as f:
        for img in data:
            date = int(img.split('_')[1][-4:])
            if date < splitdate:
                f.write(img + '\n')

allfile = ['refine_train.txt', 'refine_test.txt', 'pose_train_filter.txt', 'pose_test_filter.txt']
out_file = '1113_test.txt'

def extract(allfile, out_file, splitdate):
    allimgs = []
    for file in allfile:
        with open(file) as f:
            data = [x.strip() for x in f.readlines()]
        for img in data:
            date = int(img.split('_')[1][-4:])
            if date > splitdate:
                allimgs.append(img)
    with open(out_file, 'w') as f:
        f.write('\n'.join(allimgs))

if __name__ == '__main__':
    #split(infile, outfile, splitdate)
    extract(allfile, out_file, splitdate)
