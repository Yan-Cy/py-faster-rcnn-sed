import os
import shutil
import cv2
import toolkit

miss_file = '/home/chenyang/sed/data/box_miss.txt'
video_dir = '/home/chenyang/sed/data/videos/'
output_dir = '/home/chenyang/sed/data/missing_img'

def get_missing_box():
    extract_file = []
    with open(miss_file) as f:
        videos = dict()
        for img_name in f:
            img_name = img_name.strip()
            data = img_name.split('_')
            sub_dir = '_'.join(data[:4])
            # img_path = os.path.join(root_dir, sub_dir, data[-1] + '.jpg')
            #print img_path
            # if os.path.exists(img_path):
                # shutil.copyfile(img_path, output_dir + img_name + '.jpg')
            #else:
                #print 'Can\'t find file ' + img_path
                #extract_file.append(img_name)
            '''   
            date = int(data[1][-4:])
            if date <= 1112:
                video_path = os.path.join(video_dir, 'Dev08-1')
            else:
                video_path = os.path.join(video_dir, 'Eev08-1')
            video_file = os.path.join(video_path, sub_dir + '.avi')
            if video_file not in videos:
                videos[video_file] = 1
                if not os.path.exists(video_file):
                    print 'Can\'t find video file', video_file
                    continue
                print 'Copying', video_file
                shutil.copyfile(video_file, output_dir + sub_dir + '.avi')
            '''
            video_file = os.path.join(video_dir, sub_dir + '.avi')
            cap = cv2.VideoCapture(video_file)
            frameCnt = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            frame = int(data[-1])
            print frame, frameCnt
            if frame > frameCnt:
                print 'frame overflow:', video_file, frame
                continue
            toolkit.skipFrame(cap, frame)
            flag, img = cap.read()
            outfile = os.path.join(output_dir, img_name + '.jpg')
            cv2.imwrite(outfile, img)
            print 'frame write:', img_name, frame
            cap.release()

anno_dir = '/home/chenyang/sed/data/Annotations/raw_bbox/'
train_set = '/home/chenyang/lib/ImageSets/roi_train.txt'
test_set = '/home/chenyang/lib/ImageSets/roi_test.txt'

def remove_missing_box():
    annos = os.listdir(anno_dir)
    with open(miss_file) as f:
        miss_imgs = f.read().splitlines() 
    with open(train_set, 'w') as train, open(test_set, 'w') as test:
        for anno in annos:
            name, ext = os.path.splitext(anno)
            if name in miss_imgs:
                continue
            data = name.split('_')
            date = int(data[1][-4:])
            if date > 1201:
                test.write(name + '\n')
            else:
                train.write(name + '\n')




if __name__ == '__main__':
    # get_missing_box()
    remove_missing_box() 
