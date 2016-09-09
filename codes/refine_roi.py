import os

root = '/home/chenyang/refine/'
classes = ['CellToEar', 'Pointing', 'Embrace']
roi_path = '/home/chenyang/sed/data/Annotations/refine/'

threhold = 0.8
def same_roi(roi_a, roi_b):
    #if roi_a != roi_b:
        #print 'a', roi_a
        #print 'b', roi_b
    if roi_a[0] != roi_b[0]:
        return False
    ax0 = int(roi_a[1])
    ay0 = int(roi_a[2])
    ax1 = int(roi_a[3])
    ay1 = int(roi_a[4])

    bx0 = int(roi_b[1])
    by0 = int(roi_b[2])
    bx1 = int(roi_b[3])
    by1 = int(roi_b[4])

    if ax0 <= bx0 and ax1 >= bx1 and ay0 <= by0 and ay1 >= by1:
        return True
    if bx0 <= ax0 and bx1 >= ax1 and by0 <= ay0 and by1 >= ay1:
        return True

    Intersect = (min(ax1, bx1) - max(ax0,bx0)) * (min(ay1, by1) - max(ay0,by0))
    if Intersect < 0:
        return False
    Union = (max(ax1, bx1) - min(ax0,bx0)) * (max(ay1, by1) - min(ay0,by0))

    #print float(Intersect) / float(Union)
    return float(Intersect) / float(Union) > threhold

def check_dupl(roi_file, new_roi):
    with open(roi_file) as f:
        rois = [x.strip().split(' ') for x in f.readlines()]
    for roi in rois:
        if same_roi(roi, new_roi):
            return True
    return False

def refine_roi(src_path, roi_path):
    files = os.listdir(src_path)
    total = 0
    count = 0
    for src in files:
        total = total + 1
        name, ext = os.path.splitext(src)
        data = name.split('_')
        img_name = '_'.join(data[:5])

        roi_file = os.path.join(roi_path, img_name + '.roi')
        #print roi_file
        if os.path.exists(roi_file):
            if check_dupl(roi_file, data[5:]):
                continue
            print roi_file
            with open(roi_file, 'a') as f:
                f.write(' '.join(data[5:]) + '\n')
        else:
            with open(roi_file, 'w') as f:
                f.write(' '.join(data[5:]) + '\n')
        count = count + 1
    print 'Total:', total
    print 'Roi written:', count


train_set = '/home/chenyang/lib/ImageSets/refine_train.txt'
test_set = '/home/chenyang/lib/ImageSets/refine_test.txt'
def generate_imgset():
    print roi_path
    annos = os.listdir(roi_path)
    with open(train_set, 'w') as train, open(test_set, 'w') as test:
        total_test = 0
        total_train = 0
        for anno in annos:
            name, ext = os.path.splitext(anno)
            data = name.split('_')
            date = int(data[1][-4:])
            if date > 1201:
                total_test = total_test + 1
                test.write(name + '\n')
            else:
                total_train = total_train + 1
                train.write(name + '\n')
        print 'Total train: ', total_train
        print 'Total test: ', total_test


if __name__ == '__main__':
    for cls in classes:
        print cls
        src_path = os.path.join(root, cls, 'refined')
        refine_roi(src_path, roi_path)
    generate_imgset()
    
