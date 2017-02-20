import os
import cv2

allfile = os.listdir('.')

for gtf in allfile:
    if gtf[-4:] != '.xml':
        continue

    video = os.path.join('/home/chenyang/cydata/jiac/sed', gtf[:-4] + '.mpeg')
    cap = cv2.VideoCapture(video)
    length = str(int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))

    print gtf, length
    with open(gtf) as f:
        with open('/home/chenyang/lib/evaluation/gtf_template_new/' + gtf, 'w') as fout:
            for line in f:
                fout.write(line.replace('188832', length))
