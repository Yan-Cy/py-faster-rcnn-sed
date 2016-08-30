import os
import shutil
#import cv2
#import toolkit

miss_file = '/home/yangc1/box_miss.txt'
video_dir = '/home/yangc1/SED.raw/video/'
output_dir = '/home/yangc1/videos/'

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
            cap = cv2.VideoCapture(video_file)
            frameCnt = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            frame = int(data[-1])
            if frame > frameCnt:
                print 'frame overflow:', video_file, frame
                continue
            toolkit.skipFrame(cap, frame)
            flag, img = cap.read()
            outfile = os.path.join(output_dir, img_name + '.jpg')
            cv2.imwrite(outfile, img)
            print 'frame write:', img_name, frame
            cap.release()
            '''


if __name__ == '__main__':
    get_missing_box()
     
