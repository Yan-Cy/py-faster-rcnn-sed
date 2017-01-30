function run_CPM(images, rects)
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing/src'); 
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing/util');
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing/util/ojwoodford-export_fig-5735e6d/');
param = config();

fprintf('Description of selected model: %s \n', param.model(param.modelID).description);

%% Edit this part
% put your own test image here
%test_image = 'sample_image/sed1.jpg';
%test_image = 'sample_image/sed9.jpg';
%test_image = 'sample_image/LGW_20071206_E1_CAM1_32673.jpg'
%test_image = 'sample_image/singer.jpg';
%test_image = 'sample_image/shihen.png';
%test_image = 'sample_image/roger.png';
%test_image = 'sample_image/nadal.png';
%test_image = 'sample_image/LSP_test/im1640.jpg';
%test_image = 'sample_image/CMU_panoptic/00000998_01_01.png';
%test_image = 'sample_image/CMU_panoptic/00004780_01_01.png';
%test_image = 'sample_image/FLIC_test/princess-diaries-2-00152201.jpg';
interestPart = 'Lwri'; % to look across stages. check available names in config.m


%indexfile = fopen('sample_image/selected/index.txt', 'r');
%test_images = {};
%rectangle = {};
%while true
%    newline = fgetl(indexfile);
%    if ~ischar(newline); break; end
%    test_images = [test_images; [newline]];
%
%    t = strsplit(newline, '_');
%    t = t(length(t) - 3 : length(t));
%    t4 = t{4};
%    rectangle = [rectangle; [str2num(t{1}), str2num(t{2}), str2num(t{3}), str2num(t4(1:length(t4)-4))]];
%end
%test_images
%rectangle
%fclose(indexfile)


%% core: apply model on the image, to get heat maps and prediction coordinates
%figure(1); 
%imshow(test_image);
%hold on;
%title('Drag a bounding box');
%rectangle = getrect(1);
%for i = 1:length(test_images)
[heatMaps, prediction] = applyModel(test_image, param, rectangle)

%% visualize, or extract variable heatMaps & prediction for your use
visualize(test_image, heatMaps, prediction, param, rectangle, interestPart);
%end
