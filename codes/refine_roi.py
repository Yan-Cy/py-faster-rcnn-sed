import os

root = '/home/chenyang/Results/'
add_path = '/home/chenyang/sed/data/Annotations/refine_jiac/'
roi_path = '/home/chenyang/sed/data/Annotations/refine/'

def refine_roi(src_path, roi_path):
    files = os.listdir(src_path)
    count = 0
    for src in files:
        name, ext = os.path.splitext(src)
        data = name.split('_')
        img_name = '_'.join(data[:5])

        roi_file = os.path.join(roi_path, img_name + roi)
        os.path.exists(roi_file)
        with open(roi_file, 'a') as f:
            f.write(' '.join(data[5:]))
        count = count + 1
    print 'Roi written:', count

if __name__ == '__main__':
    for cls in classes:
        src_path = os.path.join(root, cls, 'refined')
        refine_roi(src_path, roi_path)
