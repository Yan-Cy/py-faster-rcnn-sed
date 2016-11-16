import os
import shutil
import re
import collections

cams = ['LGW_20071123_E1_CAM1', 'LGW_20071123_E1_CAM2', 'LGW_20071123_E1_CAM3', 'LGW_20071123_E1_CAM4', 'LGW_20071123_E1_CAM5', 
        'LGW_20071130_E1_CAM1', 'LGW_20071130_E1_CAM2', 'LGW_20071130_E1_CAM3', 'LGW_20071130_E1_CAM4', 'LGW_20071130_E1_CAM5', 
        'LGW_20071130_E2_CAM1', 'LGW_20071130_E2_CAM2', 'LGW_20071130_E2_CAM3', 'LGW_20071130_E2_CAM4', 'LGW_20071130_E2_CAM5',
        'LGW_20071206_E1_CAM1', 'LGW_20071206_E1_CAM2', 'LGW_20071206_E1_CAM3', 'LGW_20071206_E1_CAM4','LGW_20071206_E1_CAM5', 
        'LGW_20071207_E1_CAM2', 'LGW_20071207_E1_CAM3', 'LGW_20071207_E1_CAM4', 'LGW_20071207_E1_CAM5']

raw_data = '/mnt/sdc/chenyang/sed/Eve08/'
img_db = '/mnt/sdc/chenyang/sed/data/Images/'

def prepare_db():
    ftest = open('1113test.txt', 'w')
    for cam in cams:
        count = 0
        data_path = os.path.join(raw_data, cam)
        imgs = os.listdir(data_path)
        for img in imgs:
            imgname = cam + '_' + os.path.splitext(img)[0]
            src = os.path.join(data_path, img)
            dst = os.path.join(img_db, imgname + '.jpg')
            if imgname[-1] != '5':  # choose img each 10 frame
                continue
            #print imgname, src, dst
            count = count + 1
            shutil.copy(src, dst)
            #ftest.write(imgname + '\n')
        print 'Images for {}: {}'.format(cam, str(count))

def run_exp(dectfiles, dirname):
    os.chdir('/home/chenyang/lib/evaluation')

    ecf_head = '<ecf xmlns="http://www.itl.nist.gov/iad/mig/tv08ecf#">\n\t<source_signal_duration>64800.0</source_signal_duration>\n\t<version>20090709</version>\n\t<excerpt_list>\n'
    ecf_tail = '\t</excerpt_list>\n</ecf>\n'
    ecf_temp = '\t\t<excerpt>\n\t\t\t<filename>{}</filename>\n\t\t\t<begin>0.0</begin>\n\t\t\t<duration>7200.0</duration>\n\t\t\t<language>english</language>\n\t\t\t<source_type>surveillance</source_type>\n\t\t\t<sample_rate>25.0</sample_rate>\n\t\t</excerpt>\n'
    ecf_content = ''
    for dectfile in dectfiles:
        ecf_content = ecf_content + ecf_temp.format(dectfile + '.mpeg')
    ecf_content = ecf_head + ecf_content + ecf_tail
    with open('template/TRECVid08_ecf_v2/ecf.xml', 'w') as f:
        f.write(ecf_content)

    outdir = 'output/' + dirname + '/'
    print outdir
    if os.path.isdir(outdir):
        os.system('rm -r ' + outdir)
    os.mkdir(outdir)
    os.mkdir(outdir + 'xml')
    shutil.copy('eval.sh', outdir)
    os.chdir(outdir)
    os.system('pwd')
    os.system('ls')
    os.system('chmod +x eval.sh')
    os.system('./eval.sh')

CLASSES = ['Embrace', 'Pointing', 'CellToEar']
dettemplate = '/home/chenyang/sed/results/comp4_0b999a0c-87e3-4ec8-8d89-ffe5df1805bd_det_test_{}.txt'
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
                
                if imgdet[0] > 188832:
                    break

                if imgdet[0] == right:
                    continue
                if imgdet[0] - right < 30:
                    right = imgdet[0]
                    total = total + imgdet[1]
                    count = count + 1
                else:
                    if right - left > 20:
                        segment = '%d:%d'%(left, right)
                        id = id + 1
                        score = total * 1.0 / count
                        detcsv[imgname].append([id, cls, segment, score, score > threshold])
                    
                    left = imgdet[0]
                    right = left
                    total = imgdet[1]
                    count = 1
    
    os.system('rm -r csv && mkdir csv')
    os.system('rm -r xml && mkdir xml')
    
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

    os.system('chmod +x gen.sh')
    os.system('./gen.sh')


def prepare_gtf():
    os.system('rm -r gtf_csv && mkdir gtf_csv')
    os.system('rm -r gtf_xml && mkdir gtf_xml')

    gtf_path = '/home/chenyang/sed/Eve08/gtf/'
    gtxml_path = 'gtf_csv/'
    gtfs = cams 

    for gtf in gtfs:
        print gtf
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

    os.system('chmod +x gtf_gen.sh')
    os.system('./gtf_gen.sh')

def exp_control():
    queue = [['LGW_20071123_E1_CAM1'], 
             ['LGW_20071123_E1_CAM2'],
             ['LGW_20071123_E1_CAM5'],
             ['LGW_20071130_E1_CAM1'],
             ['LGW_20071130_E1_CAM2'],
             ['LGW_20071130_E1_CAM3'],
             ['LGW_20071130_E1_CAM5'],
             ['LGW_20071130_E2_CAM1'],
             ['LGW_20071130_E2_CAM2'],
             ['LGW_20071130_E2_CAM5'],
             ['LGW_20071206_E1_CAM1'], 
             ['LGW_20071206_E1_CAM2'], 
             ['LGW_20071206_E1_CAM3'], 
             ['LGW_20071206_E1_CAM5'], 
             ['LGW_20071207_E1_CAM2'],
             ['LGW_20071207_E1_CAM3'], 
             ['LGW_20071207_E1_CAM5'],
            ]
    dirnames = [x[0] for x in queue]

    #queue = [['LGW_20071206_E1_CAM1', 'LGW_20071206_E1_CAM2', 'LGW_20071206_E1_CAM3', #'LGW_20071206_E1_CAM4', 
    #        'LGW_20071206_E1_CAM5', 'LGW_20071207_E1_CAM2', 'LGW_20071207_E1_CAM3', #'LGW_20071207_E1_CAM4',
    #        'LGW_20071207_E1_CAM5']]
    #dirnames = ['1201']

    queue = [
            ['LGW_20071123_E1_CAM1', 'LGW_20071130_E1_CAM1', 'LGW_20071130_E2_CAM1', 'LGW_20071206_E1_CAM1'],
            ['LGW_20071123_E1_CAM1', 'LGW_20071130_E1_CAM2', 'LGW_20071130_E2_CAM2', 'LGW_20071206_E1_CAM2', 'LGW_20071207_E1_CAM2'],
            ['LGW_20071130_E1_CAM3', 'LGW_20071206_E1_CAM3', 'LGW_20071207_E1_CAM3'],
            ['LGW_20071123_E1_CAM5', 'LGW_20071130_E1_CAM5', 'LGW_20071130_E2_CAM5', 'LGW_20071206_E1_CAM5', 'LGW_20071207_E1_CAM5']
            ]
    dirnames = ['CAM1', 'CAM2', 'CAM3', 'CAM5']

    queue = [['LGW_20071123_E1_CAM1', 'LGW_20071123_E1_CAM2', 'LGW_20071123_E1_CAM5', 
        'LGW_20071130_E1_CAM2', 'LGW_20071130_E1_CAM3', 'LGW_20071130_E1_CAM5', 
        'LGW_20071130_E2_CAM1', 'LGW_20071130_E2_CAM2', 'LGW_20071130_E2_CAM5',
        'LGW_20071206_E1_CAM2', 'LGW_20071206_E1_CAM3', 'LGW_20071206_E1_CAM5', 
        'LGW_20071207_E1_CAM2', 'LGW_20071207_E1_CAM3', 'LGW_20071207_E1_CAM5'
        ]]
    dirnames = ['All_refine']

    for ind, dectfiles in enumerate(queue):
        dirname = dirnames[ind]
        run_exp(dectfiles, dirname)
    

if __name__ == '__main__':
    #prepare_db()
    prepare_csv()
    xml_script()
    #prepare_gtf()
    #gtf_script()
    exp_control()

