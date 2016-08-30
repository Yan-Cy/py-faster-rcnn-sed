import os
import json
import itertools
import tarfile
import yaml
import bisect

import cv2

import toolkit

'''func
'''

'''expr
'''
def transform_box2json():
  root_dir = '/data/MM21/SED/box/'
  outfile = '/home/jiac/data/sed.raw/box_label/train.label.json'

  names = os.listdir(root_dir)
  out = []
  for name in names:
    file = root_dir + name
    name, ext = os.path.splitext(name)
    data = name.split('_')
    date = data[1]
    E = data[2]
    camera = data[3]
    event = data[4]
    begin = int(data[5])
    end = int(data[6])
    second_in_frame = data[7]

    with open(file) as f:
      lines = f.readlines()
      lines = lines[2:]
      data = ''.join(lines)
      d = yaml.load(data)
      label = d['annotation']['object']

    out.append({
      'date': date,
      'camera': camera,
      'E': E,
      'event': event,
      'begin': begin,
      'end': end,
      'second_in_frame': second_in_frame,
      'label': label
    })
  json.dump(out, open(outfile, 'w'), indent=2)


def link_box_img():
  root_dir = '/home/jiac/data/sed.raw/'
  events = [
    'CellToEar',
    'Embrace',
    'ObjectPut',
    'PeopleMeet',
    'PeopleSplitUp',
    'PersonRuns',
    'Pointing',
  ]
  outdir = root_dir + 'img_label'

  for event in events:
    eventdir = os.path.join(root_dir, 'box_label', event)
    subdir_names = os.listdir(eventdir)
    for subdir_name in subdir_names:
      shotdir = os.path.join(eventdir, subdir_name, 'imgs')
      shotnames = os.listdir(shotdir)
      for shotname in shotnames:
        imgdir = os.path.join(shotdir, shotname)
        imgnames = os.listdir(imgdir)
        for imgname in imgnames:
          srcfile = os.path.join(imgdir, imgname)
          pos = shotname.find('.')
          data = shotname[pos+5:].split('_')
          fields = [shotname[:pos], data[1], data[0].replace('-', '_'), imgname]
          new_imgname = '_'.join(fields)
          dstfile = os.path.join(outdir, new_imgname)
          os.symlink(srcfile, dstfile)


# there exists more than one label per image
def check_box_info():
  root_dir = '/home/chenjia/hdd/data/sed/2016/'
  label_file = root_dir + 'dev/box_label/train.label.json'

  data = json.load(open(label_file))

  # label_num = dict()
  # for d in data:
  #   labels = d['label']
  #   num = len(labels)
  #   if num not in label_num:
  #     label_num[num] = 0
  #   label_num[num] += 1
  # print label_num

  pose_cnt = dict()
  name_cnt = dict()
  for d in data:
    labels = d['label']
    for label in labels:
      pose = label['pose']
      name = label['name']
      if pose not in pose_cnt:
        pose_cnt[pose] = 0
      pose_cnt[pose] += 1
      if name not in name_cnt:
        name_cnt[name] = 0
      name_cnt[name] += 1
  print pose_cnt
  print name_cnt


def split_bounding_box():
  root_dir = '/home/chenjia/hdd/data/sed/2016/'
  label_file = root_dir + 'dev/box_label/train.label.json'
  outdir = root_dir + 'dev/box_label/'

  type_camera2event_data = dict()
  data = json.load(open(label_file))
  for d in data:
    eventtype = d['event']
    camera = d['camera']
    date = d['date']
    E = d['E']
    begin = d['begin']
    end = d['end']
    second_in_frame = d['second_in_frame']
    labels = d['label']

    videoname = '_'.join([eventtype, E, date, camera])
    event = '%d_%d'%(begin, end)
    imgname = '_'.join([
      'LGW', d['date'], E, camera, eventtype, 
      str(begin), str(end), str(second_in_frame)
    ])

    if videoname not in type_camera2event_data:
      type_camera2event_data[videoname] = {}
    if event not in type_camera2event_data[videoname]:
      type_camera2event_data[videoname][event] = []
    type_camera2event_data[videoname][event].append({
      'img': imgname,
      'second_in_frame': second_in_frame,
      'label': labels
    })

  for videoname in type_camera2event_data:
    outsubdir = os.path.join(outdir, videoname)
    if not os.path.exists(outsubdir):
      os.mkdir(outsubdir)
    event_data = type_camera2event_data[videoname]
    out = []
    for event in event_data:
      data = event_data[event]
      outfile = os.path.join(outsubdir, event + '.json')
      data = sorted(data, key=lambda x:int(x['second_in_frame']))
      json.dump(data, open(outfile, 'w'))

      out.append(event)
    outfile = os.path.join(outdir, videoname + '.json')
    json.dump(out, open(outfile, 'w'))

  videos = type_camera2event_data.keys()
  outfile = os.path.join(outdir, 'event_date_camera.json')
  json.dump(videos, open(outfile, 'w'))


def gen_box_meta_json():
  root_dir = '/home/chenjia/hdd/data/sed/'
  outfile = os.path.join(root_dir, 'raw', 'box_label', 'meta.json')

  names = os.listdir(os.path.join(root_dir, 'raw', 'box_label'))
  event2names = dict()
  for name in names:
    if 'json' in name and name != 'train.label.json' and name != 'meta.json':
      data = name.split('_')
      event = data[0]
      if event not in event2names: event2names[event] = []
      event2names[event].append(name)

  json.dump(event2names.items(), open(outfile, 'w'))


def tst_draw_box():
  imgfile = '/tmp/LGW_20071101_E1_CAM1_PeopleMeet_115041_115084_0.jpg'
  outfile = '/tmp/box.jpg'

  xmin = 184
  xmax = 214
  ymin = 60
  ymax = 148

  img = cv2.imread(imgfile)
  cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
  cv2.imwrite(outfile, img)


def bat_draw_box():
  root_dir = '/home/jiac/data/sed.raw/'
  labelfile = root_dir + 'box_label/train.label.json'
  imgdir = root_dir + 'img_label/'
  outdir = root_dir + 'box_label/viz/'
  
  data = json.load(open(labelfile))
  cnt = 0
  for d in data:
    eventtype = d['event']
    E = d['E']
    camera = d['camera']
    begin = d['begin']
    end = d['end']
    second_in_frame = d['second_in_frame']
    labels = d['label']

    imgname = '_'.join([
      'LGW', d['date'], E, camera, eventtype, 
      str(begin), str(end), str(second_in_frame)
    ])
    imgfile = os.path.join(imgdir, imgname + '.jpg')
    outfile = os.path.join(outdir, imgname + '.jpg')

    img = cv2.imread(imgfile)
    for label in labels:
      bndbox = label['bndbox']
      xmin = int(bndbox['xmin'])
      xmax = int(bndbox['xmax'])
      ymin = int(bndbox['ymin'])
      ymax = int(bndbox['ymax'])
      cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    cv2.imwrite(outfile, img)

    cnt += 1
    if cnt % 1000 == 0: print cnt


def event_label_cnt():
  root_dir = '/home/chenjia/hdd/data/sed/2016/'
  label_file = os.path.join(root_dir, 'dev', 'box_label', 'train.label.json')
  outfile = os.path.join(root_dir, 'dev', 'box_label.stat.txt')

  events = [
    'CellToEar',
    'Embrace',
    'ObjectPut',
    'PeopleMeet',
    'PeopleSplitUp',
    'PersonRuns',
    'Pointing',
  ]

  event2cnt = dict()
  for event in events:
    event2cnt[event] = 0

  data = json.load(open(label_file))
  for d in data:
    event = d['event']
    event2cnt[event] += 1

  with open(outfile, 'w') as fout:
    for event in event2cnt:
      fout.write('%s %d\n'%(event, event2cnt[event]))


def extract_label_part():
  root_dir = '/home/jiac/data/sed.raw/'
  labelfile = os.path.join(root_dir, 'box_label', 'train.label.json')
  img_rootdir = os.path.join(root_dir, 'image_pos_unique')
  outdir = os.path.join(root_dir, 'box_label_part_img')

  # events = set(['Embrace', 'Pointing',])
  events = set(['CellToEar'])

  data = json.load(open(labelfile))
  cnt = 0
  for d in data:
    event = d['event']
    date = d['date']
    E = d['E']
    camera = d['camera']
    begin = d['begin']
    second_in_frame = int(d['second_in_frame'])

    if event not in events: continue

    video = '_'.join(['LGW', date, E, camera])
    imgdir = os.path.join(img_rootdir, video)
    frame = begin + second_in_frame
    imgfile = os.path.join(imgdir, '%d.jpg'%frame)

    if not os.path.exists(imgfile):
      print 'file not exists: ' + imgfile
      continue

    img = cv2.imread(imgfile)
    if img is None:
      print 'invalid img: ' + imgfile
      continue

    labels = d['label']
    for l, label in enumerate(labels):
      bndbox = label['bndbox']
      xmin = int(bndbox['xmin'])
      xmax = int(bndbox['xmax'])
      ymin = int(bndbox['ymin'])
      ymax = int(bndbox['ymax'])

      xmin = int(xmin / 320.0 * 720)
      xmax = int(xmax / 320.0 * 720)
      ymin = int(ymin / 240.0 * 576)
      ymax = int(ymax / 240.0 * 576)

      img_part = img[ymin:ymax+1, xmin:xmax+1, :]
      outfile = os.path.join(outdir, event, '%s_%d_%d.jpg'%(video, frame, l))
      if os.path.exists(outfile): continue
      print 'write img: ' + imgfile
      cv2.imwrite(outfile, img_part)

    cnt += 1


def tst_extract_label_part():
  root_dir = '/tmp/LGW_20071130_E1_CAM3/'
  imgfiles = [
    os.path.join(root_dir, '113074.jpg'),
    os.path.join(root_dir, '115638.jpg'),
  ]
  outfiles = [
    os.path.join('/tmp/eval', '113074.jpg'),
    os.path.join('/tmp/eval', '115638.jpg'),
  ]
  boxes = [
    [2, 42, 119, 180],
    [254, 302, 128, 221],
  ]
  for i in range(2):
    imgfile = imgfiles[i]
    outfile = outfiles[i]
    box = boxes[i]

    img = cv2.imread(imgfile)
    xmin = int(box[0] / 320.0 * 720)
    xmax = int(box[1] / 320.0 * 720)
    ymin = int(box[2] / 240.0 * 576)
    ymax = int(box[3] / 240.0 * 576)

    img_part = img[ymin:ymax+1, xmin:xmax+1, :]
    cv2.imwrite(outfile, img_part)


def missing_label_part():
  root_dir = '/home/jiac/data/sed.raw/'
  labelfile = os.path.join(root_dir, 'box_label', 'train.label.json')
  img_rootdir = os.path.join(root_dir, 'image_pos_unique')
  outfile = os.path.join(root_dir, 'box_label_part_img', 'missing.embrace.pointing.lst')

  events = set(['Embrace', 'Pointing',])
  # events = set(['CellToEar'])

  data = json.load(open(labelfile))
  with open(outfile, 'w') as fout:
    for d in data:
      event = d['event']
      date = d['date']
      E = d['E']
      camera = d['camera']
      begin = d['begin']
      second_in_frame = int(d['second_in_frame'])

      if event not in events: continue

      video = '_'.join(['LGW', date, E, camera])
      imgdir = os.path.join(img_rootdir, video)
      frame = begin + second_in_frame
      imgfile = os.path.join(imgdir, '%d.jpg'%frame)

      if not os.path.exists(imgfile):
        fout.write(imgfile + '\n')
      else:
        im = cv2.imread(imgfile)
        if im is None:
          fout.write(imgfile + '\n')


def extract_missing_imgs():
  root_dir = '/home/jiac/data/sed.raw/'
  # missing_lstfile = os.path.join(root_dir, 'box_label_part_img', 'missing.lst')
  missing_lstfile = os.path.join(root_dir, 'box_label_part_img', 'missing.embrace.pointing.lst')
  out_root_dir = os.path.join(root_dir, 'image_pos_unique')

  video2frames = dict()
  with open(missing_lstfile) as f:
    for line in f:
      line = line.strip()
      data = line.split('/')
      video = data[-2]
      name, ext = os.path.splitext(data[-1])
      frame = int(name)

      if video not in video2frames:
        video2frames[video] = []
      video2frames[video].append(frame)

  for video in video2frames:
    frames = video2frames[video]
    frames = sorted(frames)
    data = video.split('_')
    date = int(data[-3][-4:])
    if date <= 1112:
      video_root_dir = os.path.join(root_dir, 'video', 'Dev08-1')
    else:
      video_root_dir = os.path.join(root_dir, 'video', 'Eev08-1')

    videofile = os.path.join(video_root_dir, video + '.avi')
    out_sub_dir = os.path.join(out_root_dir, video)

    cap = cv2.VideoCapture(videofile)
    frameCnt = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    # assert frames[-1] < frameCnt
    prev = 0
    for frame in frames:
      if frame > frameCnt:
        print 'frame overflow:', video, frame
        break
      toolkit.skipFrame(cap, frame-prev)
      flag, img = cap.read()
      outfile = os.path.join(out_sub_dir, '%d.jpg'%frame)
      cv2.imwrite(outfile, img)
      print 'frame write:', video, frame

      prev = frame+1
    cap.release()


def tst_extract_missing_imgs():
  root_dir = '/media/chenjia/TOSHIBA EXT/data/sed/'
  missing_lstfile = '/tmp/missing.lst'
  out_root_dir = '/tmp/'

  video2frames = dict()
  with open(missing_lstfile) as f:
    for line in f:
      line = line.strip()
      data = line.split('/')
      video = data[-2]
      name, ext = os.path.splitext(data[-1])
      frame = int(name)
      if video != 'LGW_20071130_E1_CAM3': continue

      if video not in video2frames:
        video2frames[video] = []
      video2frames[video].append(frame)

  for video in video2frames:
    frames = video2frames[video]
    frames = sorted(frames)
    data = video.split('_')
    date = int(data[-3][-4:])
    if date <= 1112:
      video_root_dir = os.path.join(root_dir, 'video', 'Dev08-1')
    else:
      video_root_dir = os.path.join(root_dir, 'video', 'Eev08-1')

    videofile = os.path.join(video_root_dir, video + '.avi')
    out_sub_dir = os.path.join(out_root_dir, video)
    if not os.path.exists(out_sub_dir): os.mkdir(out_sub_dir)

    cap = cv2.VideoCapture(videofile)
    prev = 0
    for frame in frames:
      toolkit.skipFrame(cap, frame-prev)
      flag, img = cap.read()
      outfile = os.path.join(out_sub_dir, '%d.jpg'%frame)
      cv2.imwrite(outfile, img)
      print video, frame

      prev = frame+1
    cap.release()


def check_label_part_info_and_img():
  root_dir = '/home/jiac/data/sed.raw/'
  labelfile = os.path.join(root_dir, 'box_label', 'train.label.json')
  img_rootdir = os.path.join(root_dir, 'image_pos_unique')
  outfile = os.path.join(root_dir, 'box_label_part_img', 'failed.lst')
  
  events = set(['Embrace', 'Pointing'])

  data = json.load(open(labelfile))
  embrace_event_cnt = 0
  pointing_event_cnt = 0
  embrace_img_cnt = 0
  pointing_img_cnt = 0
  embrace_valid_img_cnt = 0
  pointing_valid_img_cnt = 0
  # embrace_unique_imgs = set()
  # pointing_unique_imgs = set()
  with open(outfile, 'w') as fout:
    for d in data:
      event = d['event']
      date = d['date']
      camera = d['camera']
      begin = d['begin']
      second_in_frame = int(d['second_in_frame'])

      if event not in events: continue

      if event == 'Embrace':
        embrace_event_cnt += 1
      if event == 'Pointing':
        pointing_event_cnt += 1

      video = '_'.join(['LGW', date, 'E1', camera])
      imgdir = os.path.join(img_rootdir, video)
      if not os.path.exists(imgdir):
        video = '_'.join(['LGW', date, 'E2', camera])
        imgdir = os.path.join(img_rootdir, video)

      frame = begin + second_in_frame
      imgfile = os.path.join(imgdir, '%d.jpg'%frame)

      if os.path.exists(imgfile):
        if event == 'Embrace':
          embrace_img_cnt += 1
          # embrace_unique_imgs.add(imgfile)
        else:
          pointing_img_cnt += 1
          # pointing_unique_imgs.add(imgfile)

        img = cv2.imread(imgfile)
        if img is not None: 
          if event == 'Embrace':
            embrace_valid_img_cnt += 1
          else:
            pointing_valid_img_cnt += 1
        else:
          fout.write(imgfile + '\n')

  print embrace_event_cnt, embrace_img_cnt, embrace_valid_img_cnt
  print pointing_event_cnt, pointing_img_cnt, pointing_valid_img_cnt


def tar_label_img():
  root_dir = '/home/jiac/data/sed.raw/'
  labelfile = os.path.join(root_dir, 'box_label', 'train.label.json')
  img_rootdir = os.path.join(root_dir, 'image_pos_unique')
  outfile = os.path.join(root_dir, 'box_label_part_img', 'Embrace.Pointing.tar.gz')

  events = set(['Embrace', 'Pointing',])

  data = json.load(open(labelfile))
  with tarfile.open(outfile, 'w:gz') as fout:
    for d in data:
      event = d['event']
      date = d['date']
      camera = d['camera']
      begin = d['begin']
      second_in_frame = int(d['second_in_frame'])

      if event not in events: continue

      video = '_'.join(['LGW', date, 'E1', camera])
      imgdir = os.path.join(img_rootdir, video)
      if not os.path.exists(imgdir):
        video = '_'.join(['LGW', date, 'E2', camera])
        imgdir = os.path.join(img_rootdir, video)
      frame = begin + second_in_frame
      imgfile = os.path.join(imgdir, '%d.jpg'%frame)

      fields = imgfile.split('/')
      arcname = '_'.join(fields[-2:])
      fout.add(imgfile, arcname=arcname)


def gen_label_part_img_json():
  root_dir = '/usr0/home/jiac/data/sed/box_label_part_img/'

  # events = ['Embrace', 'Pointing']
  events = ['CellToEar']

  for event in events:
    eventdir = os.path.join(root_dir, event)
    outfile = os.path.join(root_dir, event + '.json')

    names = os.listdir(eventdir)
    json.dump(names, open(outfile, 'w'))


def gen_roi_db():
  root_dir = '/home/jiac/data/sed.raw/'
  labelfile = os.path.join(root_dir, 'box_label', 'train.label.json')
  img_rootdir = os.path.join(root_dir, 'image_pos_unique')
  outfile = '/home/jiac/data/sed2016/box_label/roi.txt'

  events = set(['Embrace', 'Pointing',])

  data = json.load(open(labelfile))
  with open(outfile, 'w') as fout:
    for d in data:
      event = d['event']
      date = d['date']
      E = d['E']
      camera = d['camera']
      begin = d['begin']
      second_in_frame = int(d['second_in_frame'])

      if event not in events: continue

      video = '_'.join(['LGW', date, E, camera])
      imgdir = os.path.join(img_rootdir, video)
      frame = begin + second_in_frame
      imgfile = os.path.join(imgdir, '%d.jpg'%frame)

      img = cv2.imread(imgfile)
      if img is None:
        print 'invalid img: ' + imgfile
        continue

      labels = d['label']
      fout.write('%s%d.jpg %s '%(video, frame, event))
      for label in labels:
        bndbox = label['bndbox']
        xmin = int(bndbox['xmin'])
        xmax = int(bndbox['xmax'])
        ymin = int(bndbox['ymin'])
        ymax = int(bndbox['ymax'])

        xmin = int(xmin / 320.0 * 720)
        xmax = int(xmax / 320.0 * 720)
        ymin = int(ymin / 240.0 * 576)
        ymax = int(ymax / 240.0 * 576)
        fout.write('%d %d %d %d '%(xmin, xmax, ymin, ymax))
      fout.write('\n')


def gen_img_db():
  root_dir = '/home/jiac/data/sed.raw/'
  labelfile = os.path.join(root_dir, 'box_label', 'train.label.json')
  img_rootdir = os.path.join(root_dir, 'image_pos_unique')
  outdir = '/home/jiac/data/sed2016/box_label/img/'

  events = set(['Embrace', 'Pointing', 'CellToEar'])

  data = json.load(open(labelfile))
  for d in data:
    event = d['event']
    date = d['date']
    E = d['E']
    camera = d['camera']
    begin = d['begin']
    second_in_frame = int(d['second_in_frame'])

    if event not in events: continue

    video = '_'.join(['LGW', date, E, camera])
    imgdir = os.path.join(img_rootdir, video)
    frame = begin + second_in_frame
    imgfile = os.path.join(imgdir, '%d.jpg'%frame)

    if not os.path.exists(imgfile):
      print 'file not exists: ' + imgfile
      continue

    img = cv2.imread(imgfile)
    if img is None:
      print 'invalid img: ' + imgfile
      continue

    outfile = os.path.join(outdir, '%s_%d.jpg'%(video, frame))
    if not os.path.exists(outfile):
      os.symlink(imgfile, outfile)


def mix_img_per_second_with_pos():
  img_per_second_lstfile = '/home/jiac/data/sed.raw/image_persecond/img.lst'
  box_dir = '/data/MM21/SED/box/'
  # outfile = '/home/jiac/data/sed2016/video2reserve_frames.json'
  outfiles = [
    '/home/jiac/data/sed2016/video2reserve_frames.event.json',
    '/home/jiac/data/sed2016/video2reserve_frames.null.json',
  ]

  video2event_reserve_frames = dict()
  video2segments = dict()
  names = os.listdir(box_dir)
  for name in names:
    name, ext = os.path.splitext(name)
    data = name.split('_')
    video = '_'.join(data[:4])
    segment = [int(data[5]), int(data[6])]
    frame = segment[0] + int(data[7])
    segment[1] = max(segment[1], frame)
    if video not in video2segments: 
      video2segments[video] = []
      video2event_reserve_frames[video] = []
    video2segments[video].append(segment)
    video2event_reserve_frames[video].append(frame)

  json.dump(video2event_reserve_frames.items(), open(outfiles[0], 'w'))

  video2merge_segments = dict()
  for video in video2segments:
    segments = video2segments[video]
    points = []
    for segment in segments:
      points.append((segment[0], 0))
      points.append((segment[1], 1))

    points = sorted(points)

    merge_segments = []
    stack = []
    for point in points:
      if point[1] == 0:
        stack.append(point)
      else:
        start = stack.pop()
        if len(stack) == 0:
          merge_segments.append((start, point[0]))
    video2merge_segments[video] = merge_segments

  video2frames = dict()
  with open(img_per_second_lstfile) as f:
    for line in f:
      line = line.strip()
      data = line.split('/')
      video = data[-2]
      pos = data[-1].find('.')
      frame = int(data[-1][:pos])
      if video not in video2frames: video2frames[video] = []
      video2frames[video].append(frame)

  video2null_reserve_frames = dict()
  for video in video2frames:
    frames = video2frames[video]
    frames = sorted(frames)

    if video not in video2merge_segments: continue # skip CAM4, which is not interesting

    merge_segments = video2merge_segments[video]
    starts = [d[0] for d in merge_segments]

    reserve_frames = []
    for frame in frames:
      idx = bisect.bisect_left(starts, frame)
      reserve = True
      if starts[idx] == frame:
        reserve = False
      else:
        idx -= 1
        if idx >= 0:
          reserve = (merge_segments[idx][1] < frame)

      if reserve:
        reserve_frames.append(frame)
    video2null_reserve_frames[video] = reserve_frames

  json.dump(video2null_reserve_frames.items(), open(outfiles[1], 'w'))


'''main
'''
if __name__ == '__main__':
  # transform_box2json()
  # link_box_img()
  # check_box_info()
  # split_bounding_box()
  # gen_box_meta_json()
  # tst_draw_box()
  # bat_draw_box()
  # event_label_cnt()
  # extract_label_part()
  # tst_extract_label_part()
  # missing_label_part()
  # extract_missing_imgs()
  # tst_extract_missing_imgs()
  # check_label_part_info_and_img()
  # tar_label_img()
  # gen_label_part_img_json()
  # gen_roi_db()
  # gen_img_db()
  mix_img_per_second_with_pos()

