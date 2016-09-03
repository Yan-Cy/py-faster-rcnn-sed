import os

anno_path = '/home/chenyang/sed/data/Annotations/'
roi_dbs = ['raw_bbox', 'pose_roi']
final_path = '/home/chenyang/sed/data/Annotations/roi/'

def merge_roidb(roi_dbs):
    for roi_db in roi_dbs:
        roi_path = os.path.join(anno_path, roi_db)
        roi_files = os.listdir(roi_path)
        for roi_file in roi_files:
            f_roi = open(os.path.join(roi_path, roi_file))
            final_roi = os.path.join(final_path, roi_file)
            if os.path.exists(final_roi):
                print roi_file
            with open(final_roi, 'a') as f:
                f.write(f_roi.read())


if __name__ == '__main__':
    merge_roidb(roi_dbs)


