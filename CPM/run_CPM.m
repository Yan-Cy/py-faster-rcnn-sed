function [result, visible] = run_CPM(image, rect, param, net)
%{
close all;
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing'); 
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing/src'); 
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing/util');
addpath('/home/chenyang/workspace/convolutional-pose-machines-release/testing/util/ojwoodford-export_fig-5735e6d/');
param = config();

fprintf('Description of selected model: %s \n', param.model(param.modelID).description);
%}

rootpath = '/home/chenyang/cydata/sed_subset/annodata/';

[heatMaps, prediction] = applyModel([rootpath, 'images/', image, '.jpg'], param, rect, net);
result = prediction;

visible = zeros(length(result));
for part = 1 : length(result)
   response = heatMaps{end}(:,:,part);
   max_value = max(max(response));
   visible(part) = max_value;
end
