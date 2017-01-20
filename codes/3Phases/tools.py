
import os

def IoU(box1, box2, delta = 5):
    box1 = [float(x) for x in box1]
    box2 = [float(x) for x in box2]
    if box1[0]-delta < box2[0] and box1[2]+delta > box2[2] and box1[1]-delta < box2[1] and box1[3]+delta > box2[3]:
        return 1
    if box2[0]-delta < box1[0] and box2[2]+delta > box1[2] and box2[1]-delta < box1[1] and box2[3]+delta > box1[3]:
        return 1
    if box1[0] >= box2[2] or box1[2] <= box2[0] or box1[1] >= box2[3] or box1[3] <= box2[1]:
        return 0
    Intersect = min(box1[2], box2[2]) - max(box1[0], box2[0])
    if Intersect <= 0:
        Intersect = 0.0
    Intersect = Intersect * (min(box1[3], box2[3]) - max(box1[1], box2[1]))
    if Intersect <= 0:
        Intersect = 0.0
    Union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - Intersect
    assert Union >= 0, str(box1) + '\n' + str(box2) + '\n' + str(Union) + '\n' +  str(Intersect)
    return Intersect / Union

def overlap(bbox, annofile, threshold = 0.5):
    if not os.path.exists(annofile):
        return False
    with open(annofile) as f:
        boxes = [x.strip().split(' ') for x in f.readlines()]

    for box in boxes:
        if box[0] == bbox[0] and IoU(box[1:], bbox[1:]) > threshold:
            return True

    return False