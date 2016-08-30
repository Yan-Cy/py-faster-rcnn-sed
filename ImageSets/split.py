import re

path = 'no_pose.txt'   
with open(path) as f:
    data = f.read()

test_images = re.findall('LGW_200711\S+', data)
train_images = re.findall('LGW_200712\S+', data)

with open('no_pose_train.txt', 'wt') as f:
    for image in train_images:
        f.write(image + '\n')

with open('no_pose_test.txt', 'wt') as f:
    for image in test_images:
        f.write(image + '\n')
