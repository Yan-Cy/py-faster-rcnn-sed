function run()

testindex = '/home/chenyang/cydata/sed_subset/annodata/test.txt';
trainindex = '/home/chenyang/cydata/sed_subset/annodata/train.txt';
imagepath = '/home/chenyang/cydata/sed_subset/annodata/images/';
outputpath = '/home/chenyang/lib/CPM/';
control(trainindex, imagepath, outputpath, 'train');
control(testindex, imagepath, outputpath, 'test');

function control(indexfile, imagepath, outputpath, setname)

[images, rect, cls] = load_index(indexfile);
%predict_pose(images, rect, cls)
interest_layers_list = {{'conv5_2_CPM'}; {'Mconv7_stage2'}; {'Mconv7_stage3'};{'Mconv7_stage4'};{'Mconv7_stage5'};{'Mconv7_stage6'};};
for i = 1 : length(interest_layers_list)
    interest_layers = interest_layers_list{i}
    CPM_feature(images, rect, imagepath, outputpath, setname, interest_layers);
end

function [images, rect, cls] = load_index(indexfile)
index = fopen(indexfile, 'r');
images = {};
rect = {};
cls = {};
while true
    newline = fgetl(index);
    if ~ischar(newline); break; end
    
    t = strsplit(newline, ' ');
    images = [images; [t{1}]];
    cls = [cls; [t{2}]];
    
    t1 = str2num(t{3});
    t2 = str2num(t{4});
    t3 = str2num(t{5});
    t4 = str2num(t{6});
    rect = [rect; [t1, t2, t3-t1, t4-t2]];
end

fclose(index);


function predict_pose(images, rect, cls)

results = {};
visible = {};
for i = 1:length(images)
    [result, v] = run_CPM(images{i}, rect{i});
    results = [results; [result]];
    visible = [visible; [v]];
end

articulation = {'head', 'neck', 'Rsho', 'Relb', 'Rwri', ...
                'Lsho', 'Lelb', 'Lwri', ...
                'Rhip', 'Rkne', 'Rank', ...
                'Lhip', 'Lkne', 'Lank', 'bkg'};    

outputroot = '/home/chenyang/lib/CPM/results/';

for  i = 1:length(results)
    x1 = num2str(rect{i}(1));
    y1 = num2str(rect{i}(2));
    x2 = num2str(rect{i}(1) + rect{i}(3));
    y2 = num2str(rect{i}(2) + rect{i}(4));
    imgfile = fopen([outputroot, images{i}, '_', cls{i}, '_', x1, '_', y1, '_', x2, '_', y2, '.txt'], 'w');
    for j = 1:length(results{i})
        fprintf(imgfile, '%d %d %s %f\n', results{i}(j,1), results{i}(j, 2), articulation{j}, visible{i}(j));
    end
    fclose(imgfile);
end


function CPM_feature(images, rect, imagepath, outputpath, setname, interest_layers)

addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing');
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing/src');
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing/util');
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing/util/ojwoodford-export_fig-5735e6d/');
param = config();

fprintf('Description of selected model: %s \n', param.model(param.modelID).description);

model = param.model(param.modelID);
net = caffe.Net(model.deployFile, model.caffemodel, 'test');

features = [];
for i = 1:length(images)
    cnnfeature = extract_feature([imagepath, images{i}, '.jpg'], param, rect{i}, net, interest_layers);
    
    %TO DO
    % Write feature to file
    feature = [];
    for j = 1:length(cnnfeature)
        t = reshape(cnnfeature{j}, 1, []);
        feature = [feature, t];
    end
    
    features = [features; feature];
    %filename = [outputpath, images{i}, '.mat'];
    %save(filename, 'feature');
    info = [num2str(i), ' / ', num2str(length(images))];
    disp(info)
end

size(features)
filename = [outputpath, setname];
for i = 1:length(interest_layers)
    filename = [filename, '_', interest_layers{i}];
end
filename = [filename, '.mat']
save(filename, 'features');

