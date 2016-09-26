# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.sed
from sed_eval import sed_eval
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
import errno

class sed(datasets.imdb):
    def __init__(self, image_set, devkit_path):
        datasets.imdb.__init__(self, image_set)
        self._image_set = image_set
        self._devkit_path = devkit_path
        self._data_path = os.path.join(self._devkit_path, 'data')
        self._classes = ('__background__', # always index 0
                         'Embrace',
                         'Pointing',
                         'CellToEar'
                         )
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = ['.jpg', '.png']
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.sed_roidb
	self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # Specific config options
        self.config = {'cleanup'  : False,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'Devkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        for ext in self._image_ext:
            image_path = os.path.join(self._data_path, 'Images',
                                  index + ext)
            if os.path.exists(image_path):
                break
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._data_path + /ImageSets/val.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        #cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        #if os.path.exists(cache_file):
        #    with open(cache_file, 'rb') as fid:
        #        roidb = cPickle.load(fid)
        #    print '{} gt roidb loaded from {}'.format(self.name, cache_file)
        #    return roidb

        gt_roidb = [self._load_sed_annotation(index)
                    for index in self.image_index]
        #with open(cache_file, 'wb') as fid:
        #    cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        #print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        #cache_file = os.path.join(self.cache_path,
        #                          self.name + '_selective_search_roidb.pkl')

        #if os.path.exists(cache_file):
        #    with open(cache_file, 'rb') as fid:
        #        roidb = cPickle.load(fid)
        #    print '{} ss roidb loaded from {}'.format(self.name, cache_file)
        #    return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
            print len(roidb)
        #with open(cache_file, 'wb') as fid:
        #    cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        #print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(self._devkit_path,
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
               'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['all_boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        eturn the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        #cache_file = os.path.join(self.cache_path,
        #        '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
        #        format(self.name, self.config['top_k']))

        #if os.path.exists(cache_file):
        #    with open(cache_file, 'rb') as fid:
        #        roidb = cPickle.load(fid)
        #    print '{} ss roidb loaded from {}'.format(self.name, cache_file)
        #    return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        #with open(cache_file, 'wb') as fid:
        #    cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        #print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 self.name))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def sed_roidb(self):
        gt_roidb = self.gt_roidb()
        person_roidb = self.person_roidb(gt_roidb)
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, person_roidb)
        #roidb = person_roidb
        print self._image_set
        #if self._image_set != 'test':
        #    roidb = datasets.imdb.merge_roidbs(roidb, gt_roidb)
        roidb = datasets.imdb.merge_roidbs(roidb, ss_roidb)
        return roidb

    def person_roidb(self, gt_roidb):
        box_list = []
        for i in xrange(self.num_images):
            box_list.append(self._load_person_roidb(self.image_index[i]))
       
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_person_roidb(self, index):
        filename = os.path.join(self._data_path, 'Annotations', 'pose_roi',  index + '.roi')
        if not os.path.exists(filename):
            filenames = ['','','','']
            for i in xrange(4):
                filenames[i] = self._person_adjust(index, i)
                if os.path.exists(filenames[i]):
                    filename = filenames[i]
                    break
            if not os.path.exists(filename):
                print 'Can not find person file for', filename
                return np.zeros((0,4), dtype=np.uint16)
        
        with open(filename)as f:
            data = f.read()
        import re
        objs = re.findall('(\S+) (\d+) (\d+) (\d+) (\d+)', data)

        #objs = [x for x in objs if x[0] = 'Pose']
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)

        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(obj[1])
            y1 = float(obj[2])
            x2 = float(obj[3])
            y2 = float(obj[4])
            #print x1, y1, x2, y2
            boxes[ix, :] = self.sed_refine_box(x1, y1, x2, y2)
         
        return boxes

    def _person_adjust(self, index, step):
        data = index.split('_')
        frame2 = int(data[-1][-2:])
        t = ((frame2 / 25 + step) % 4) * 25
        #data[-1] = data[-1][:-3] + str(int(data[-1][-3]) + (step / 4)) + str(t)
        assert(len(data[-1]) >= 2)
        data[-1] = data[-1][:-2] + str(t)
        if t == 0:
            data[-1] = data[-1] + '0'
        #print index
        #print os.path.join(self._data_path, 'Annotations', 'pose_roi',  '_'.join(data) + '.roi')
        #assert False
        return os.path.join(self._data_path, 'Annotations', 'pose_roi',  '_'.join(data) + '.roi')

    box_height = 100
    def sed_refine_box(self, x1, y1, x2, y2):
        if y2 - y1 > self.box_height:
            y2 = y1 + self.box_height
        return [x1, y1, x2, y2]

    def _load_sed_annotation(self, index):
        """
        Load image and bounding boxes info from sed dataset.
        """
        filename = os.path.join(self._data_path, 'Annotations', 'roi', index + '.roi')

        with open(filename)as f:
            data = f.read()
        import re
        objs = re.findall('(\S+) (\d+) (\d+) (\d+) (\d+)', data)

        objs = [x for x in objs if self._class_to_ind.has_key(x[0])]

        #print objs
        num_objs = len(objs)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
            x1 = float(obj[1])
            y1 = float(obj[2])
            x2 = float(obj[3])
            y2 = float(obj[4])
            #print x1, y1, x2, y2
            file_cls = obj[0]
            cls = self._class_to_ind[file_cls]
            overlaps[ix, cls] = 1.0
            boxes[ix, :] = self.sed_refine_box(x1, y1, x2, y2)
            gt_classes[ix] = cls


        overlaps = scipy.sparse.csr_matrix(overlaps)
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_sed_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            #comp_id += '-{}'.format(os.getpid())
            comp_id += '_{}'.format(self._salt)

        # VOCdevkit/results/comp4-44503_det_test_aeroplane.txt
        #path = os.path.join(self._devkit_path, 'results', self.name, comp_id + '_')
        path = os.path.join(self._devkit_path, 'results', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'setenv(\'LC_ALL\',\'C\'); voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_sed_results_file(all_boxes)
        #self._do_matlab_eval(comp_id, output_dir)
	self._do_python_eval(output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_results_file_template(self):
        # devkit/results/comp4-44503_det_test_{%s}.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        try:
            os.mkdir(self._devkit_path + '/results')
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise e
        path = os.path.join(
            self._devkit_path,
            'results',
            filename)
        return path

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._data_path,
            'Annotations')
        imagesetfile = os.path.join(
            self._data_path,
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._devkit_path, 'annotations_cache')
        aps = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_results_file_template().format(cls)
            rec, prec, ap = sed_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

        return aps

if __name__ == '__main__':
    d = datasets.sed('train', '')
    res = d.roidb
    from IPython import embed; embed()

