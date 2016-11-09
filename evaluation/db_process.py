import os
import shutil
import re
import collections

cams = ['LGW_20071206_E1_CAM1', 'LGW_20071206_E1_CAM2', 'LGW_20071206_E1_CAM3', 'LGW_20071206_E1_CAM4', 'LGW_20071206_E1_CAM5', 
        'LGW_20071207_E1_CAM1', 'LGW_20071207_E1_CAM2', 'LGW_20071207_E1_CAM3', 'LGW_20071207_E1_CAM4', 'LGW_20071207_E1_CAM5']
raw_data = '/mnt/sdc/chenyang/raw_data/raw_per_5/'
img_db = '/mnt/sdc/chenyang/sed/data/Images/'

def prepare_db():
    ftest = open('1206test.txt', 'w')
    for cam in cams:
        data_path = os.path.join(raw_data, cam)
        imgs = os.listdir(data_path)
        for img in imgs:
            imgname = cam + '_' + os.path.splitext(img)[0]
            src = os.path.join(data_path, img)
            dst = os.path.join(img_db, imgname + '.jpg')
            if imgname[-1] != '5':
                continue
            print imgname, src, dst
            #shutil.copy(src, dst)
            ftest.write(imgname + '\n')

CLASSES = ['Embrace', 'Pointing', 'CellToEar']
dettemplate = '/home/chenyang/sed/results/comp4_2452f163-eff5-4863-9b0a-b78b6ccbe302_det_test_{}.txt'
threshold = 0.5

def prepare_csv():
    detcsv = dict()
    
    for cls in CLASSES:
        print cls
        detfile = dettemplate.format(cls)
        with open(detfile) as f:
            dets = [x.strip().split(' ') for x in f.readlines()]
        
        all_results = dict()
        for det in dets:
            data = det[0].split('_')
            imgname = '_'.join(data[:-1])
            frame = int(data[-1])
            score = float(det[1])
            x1 = det[2]
            y1 = det[3]
            x2 = det[4]
            y2 = det[5]
            
            if not all_results.has_key(imgname):
                all_results[imgname] = []
            all_results[imgname].append([frame, score, cls, x1, y1, x2, y2])

        for imgname in all_results:
            if not detcsv.has_key(imgname):
                detcsv[imgname] = []
            
            imgdets = sorted(all_results[imgname], key=lambda x: (x[0], x[1]))
            imgdets = [x for x in imgdets if x[1] > 0.1]
           
            if len(imgdets) == 0:
                continue

            left = imgdets[0][0]
            right = left
            total = imgdets[0][1]
            count = 1
            id = 0

            for imgdet in imgdets:
                assert imgdet[0] >= right
                
                if imgdet[0] == right:
                    continue
                if imgdet[0] - right < 30:
                    right = imgdet[0]
                    total = total + imgdet[1]
                    count = count + 1
                else:
                    segment = '%d:%d'%(left, right)
                    id = id + 1
                    score = total * 1.0 / count
                    detcsv[imgname].append([id, cls, segment, score, score > threshold])
                    
                    left = imgdet[0]
                    right = left
                    total = imgdet[1]
                    count = 1

    for imgname in detcsv:
        csvfile = 'csv/' + imgname + '.csv'
        with open(csvfile, 'w') as f:
            f.write('"ID","EventType","Framespan","DetectionScore","DetectionDecision"\n')
            for id, cls, segment, score, decision in detcsv[imgname]:
                f.write('"%d","%s","%s","%f","%d"\n'%(id, cls, segment, score, decision))


def xml_script():
    outfile = 'gen.sh'
    outdir = '/home/chenyang/lib/evaluation/xml/'
    detdir = '/home/chenyang/lib/evaluation/csv/'
    templatedir = os.path.join('/home/chenyang/lib/evaluation/xmltemplate/*.xml')

    with open(outfile, 'w') as f:
        empty_cmd = ['/mnt/sdc/chenyang/F4DE/TrecVid08/tools/TV08ViperValidator/TV08ViperValidator.pl',
                    '--limitto', 'CellToEar,Embrace,Pointing',
                    '--Remove', 'ALL',
                    '--write', outdir,
                    templatedir
                    ]
        f.write(' '.join(empty_cmd) + '\n')
        
        csvfiles = os.listdir(detdir)
        for csvfile in csvfiles:
            name = os.path.splitext(csvfile)[0]
            cmd = ['/mnt/sdc/chenyang/F4DE/TrecVid08/tools/TV08ViperValidator/TV08ViperValidator.pl',
                '--limitto', 'CellToEar,Embrace,Pointing',
                '--fps', '25',
                '--write', outdir,
                '--insertCSV', os.path.join(detdir, name + '.csv'),
                os.path.join(outdir, name + '.xml')]

            f.write(' '.join(cmd) + '\n')


def prepare_gtf():
    gtf_path = '/home/chenyang/workspace/raw_data/gtf/'
    gtxml_path = 'gtf_csv/'
    gtfs = ['LGW_20071206_E1_CAM1', 'LGW_20071206_E1_CAM2', 'LGW_20071206_E1_CAM3', 
            'LGW_20071206_E1_CAM4', 'LGW_20071206_E1_CAM5', 'LGW_20071207_E1_CAM2',
            'LGW_20071207_E1_CAM3', 'LGW_20071207_E1_CAM4', 'LGW_20071207_E1_CAM5']

    for gtf in gtfs:
        gtf_file = os.path.join(gtf_path, gtf + '.txt')
        with open(gtf_file) as f:
            gts = [x.strip().split(' ') for x in f.readlines()]
        
        gtf_xml = os.path.join(gtxml_path, gtf + '.csv')
        with open(gtf_xml, 'w') as f:
            f.write('"ID","EventType","Framespan"\n')
            id = 0
            for gt in gts:
                id = id + 1
                f.write('"%d","%s","%s:%s"\n'%(id, gt[3], gt[1], gt[2]))


def gtf_script():
    outfile = 'gtf_gen.sh'
    outdir = '/home/chenyang/lib/evaluation/gtf_xml/'
    detdir = '/home/chenyang/lib/evaluation/gtf_csv/'
    templatedir = os.path.join('/home/chenyang/lib/evaluation/gtf_template/*.xml')

    with open(outfile, 'w') as f:
        empty_cmd = ['/mnt/sdc/chenyang/F4DE/TrecVid08/tools/TV08ViperValidator/TV08ViperValidator.pl',
                    '--limitto', 'CellToEar,Embrace,Pointing',
                    '--Remove', 'ALL',
                    '--write', outdir,
                    '--gtf',
                    templatedir
                    ]
        f.write(' '.join(empty_cmd) + '\n')
        
        csvfiles = os.listdir(detdir)
        for csvfile in csvfiles:
            name = os.path.splitext(csvfile)[0]
            cmd = ['/mnt/sdc/chenyang/F4DE/TrecVid08/tools/TV08ViperValidator/TV08ViperValidator.pl',
                '--limitto', 'CellToEar,Embrace,Pointing',
                '--fps', '25',
                '--write', outdir,
                '--gtf',
                '--insertCSV', os.path.join(detdir, name + '.csv'),
                os.path.join(outdir, name + '.xml')]

            f.write(' '.join(cmd) + '\n')


if __name__ == '__main__':
    #prepare_db()
    prepare_csv()
    xml_script()
    #prepare_gtf()
    #gtf_script()
