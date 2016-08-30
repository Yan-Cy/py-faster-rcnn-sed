import os
import yaml

all_events = ['Embrace', 'Pointing', 'CellToEar']
img_missing = []

def extract_bbox():
    root_dir = '/home/chenyang/sed/data/box'
    img_dir = '/home/chenyang/sed/data/Images'
    output_dir = '/home/chenyang/sed/data/raw_bbox'


    file_missed = 0
    count_event = dict()

    names = os.listdir(root_dir)
    for name in names:
        file = os.path.join(root_dir, name)
        assert os.path.exists(file)

        name, ext = os.path.splitext(name)
        data = name.split('_')
        pre = data[0]
        date = data[1]
        E = data[2]
        camera = data[3]
        event = data[4]

        if event not in count_event:
            count_event[event] = 0
        else:
            count_event[event] = count_event[event] + 1
        if event not in all_events:
            continue
        
        begin = int(data[5])
        end = int(data[6])
        second_in_frame = int(data[7])

        with open(file) as f:
            lines = f.readlines()
            lines = lines[2:]
            data = ''.join(lines)
            d = yaml.load(data)
            labels = d['annotation']['object']
    
        img_frame = begin + second_in_frame
        img_name = '_'.join([pre, date, E, camera, str(img_frame)])
        if not os.path.exists(os.path.join(img_dir, img_name + '.jpg')):
            #print 'Missing image file {}'.format(img_name)
            img_missing.append(img_name)
            file_missed = file_missed +1
        '''
        output_file = os.path.join(output_dir, img_name + '.roi')
        with open(output_file, 'a') as f:
            for label in labels:
                xmax = str(int(float(label['bndbox']['xmax']) / 320.0 * 720))
                xmin = str(int(float(label['bndbox']['xmin']) / 320.0 * 720))
                ymax = str(int(float(label['bndbox']['ymax']) / 240.0 * 576))
                ymin = str(int(float(label['bndbox']['ymin']) / 240.0 * 576))
                bbox = ' '.join([event, xmin, ymin, xmax, ymax])
                f.write(bbox)
                f.write('\n')
        '''
    for event in all_events:
        print '{}: {}'.format(event, count_event[event])
    print 'Missing Image: ' + str(file_missed)


def get_missing_img():
    log_file = '/home/chenyang/sed/data/box_miss.txt'
    with open(log_file, 'w') as f:
        f.write('\n'.join(img_missing))

    # run get_missing_img in rocks to get missing img

def check_missing_img():
    log_file = '/home/chenyang/sed/data/box_miss.txt'
    img_dir = '/home/chenyang/sed/data/Images'
    
    with open(log_file) as f:
        names = f.readlines()

    count = 0
    for name in names:
        name = name.strip()
        img_path = os.path.join(img_dir, name + '.jpg')
        if not os.path.exists(img_path):
            print 'Image {} does not exists'.format(img_path)
            count = count + 1
    
    print 'Missing Images:', count

if __name__ == '__main__':
    #extract_bbox()
    #get_missing_img()
    check_missing_img()
