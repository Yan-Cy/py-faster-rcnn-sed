from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import numpy as np
import re
import sys
import os
import pickle

annopath = '/home/chenyang/lib/CPM/results_2/'
trainanno = '/home/chenyang/cydata/sed_subset/annodata/train_annos'
testanno = '/home/chenyang/cydata/sed_subset/annodata/test_annos'
trainset = '/home/chenyang/cydata/sed_subset/annodata/train.txt'
testset = '/home/chenyang/cydata/sed_subset/annodata/test.txt'
clsname = dict()
clsname['Pose'] = 0
clsname['Embrace'] = 1
clsname['Pointing'] = 2
clsname['CellToEar'] = 3

def load_anno(setpath, annopath):
    with open(setpath) as f:
        imgset = [x.strip().split(' ') for x in f.readlines()]

    features = []
    labels = []

    #filelist = os.listdir(annopath)
    for annofile in imgset:
        annofile = '_'.join(annofile) + '.txt'
        filepath = os.path.join(annopath, annofile)
        with open(filepath) as f:
            data = [x.strip().split(' ') for x in f.readlines()]


        if len(data) != 8:
            if len(data) == 0:
                feature = [-1] * 16
                label = 0
                features.append(feature)
                labels.append(label)
                continue

            print annofile
            print data
            print 'File Format Error!'
            sys.exit()

        imgname = os.path.splitext(annofile)[0]
        imginfo = imgname.split('_')
        x1 = int(imginfo[-4])
        y1 = int(imginfo[-3])
        x2 = int(imginfo[-2])
        y2 = int(imginfo[-1])

        feature = []
        label = clsname[imginfo[-5]]

        for ind, anno in enumerate(data):
            if len(anno) != 3:
                print annofile
                print anno
                print 'File Format Error!'
                sys.exit()

            part_name = anno[2]
            x = int(anno[0])
            y = int(anno[1])

            if x != -1 and y != -1:
                x = (x - x1) * 1.0 / (x2 - x1)
                y = (y - y1) * 1.0 / (y2 - y1)
            
            feature.append(x)
            feature.append(y)


        features.append(feature)
        labels.append(label)

    return features, labels

def evaluate(pre_y, Ty):
    count = 0
    confusion = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for ind, y in enumerate(pre_y):
        if y == Ty[ind]:
            count = count + 1
        confusion[Ty[ind]][y] = confusion[Ty[ind]][y] + 1

    print 'Accuracy:', count * 1.0 / len(Ty)
    print 'Pose', confusion[0]
    print 'Embrace', confusion[1]
    print 'Pointing', confusion[2]
    print 'CellToEar', confusion[3]


if __name__ == '__main__':
    #X, Y = load_anno(trainset, annopath)
    X, Y = load_anno(trainset, trainanno)

    #cnn = pickle.load(open('/Users/chenyang/Desktop/CMU/train_features.pkl', 'rb'))
    #pca = PCA(n_components=1000)
    #pca_feature = pca.fit_transform(cnn)

    #print pca_feature
    #print len(X), pca_feature.shape

    #X = cnn
    #X = np.hstack((X, pca_feature))
    #X = np.hstack((X, cnn))

    clf = RandomForestClassifier(n_estimators=30)
    #clf = LogisticRegression(solver='sag', max_iter=100, random_state=42, multi_class='multinomial')
    clf = clf.fit(X, Y)

    #Tx, Ty = load_anno(testset, annopath)
    Tx, Ty = load_anno(testset, testanno)

    #cnn = pickle.load(open('/Users/chenyang/Desktop/CMU/test_features.pkl', 'rb'))
    #pca = PCA(n_components=1000)
    #pca_feature = pca.fit_transform(cnn)

    #Tx = cnn
    #Tx = np.hstack((Tx, pca_feature))
    #Tx = np.hstack((Tx, cnn))

    pre_y = clf.predict(Tx)
    evaluate(pre_y, Ty)

