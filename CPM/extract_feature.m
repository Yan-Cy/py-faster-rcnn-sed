function features = extract_feature(test_image, param, rectangle, net, interest_layers)
tic;
%% Select model and other parameters from param
model = param.model(param.modelID);
boxsize = model.boxsize;
np = model.np;
nstage = model.stage;
oriImg = imread(test_image);

%% Apply model, with searching thourgh a range of scales
octave = param.octave;
% set the center and roughly scale range (overwrite the config!) according to the rectangle
x_start = max(rectangle(1), 1);
x_end = min(rectangle(1)+rectangle(3), size(oriImg,2));
y_start = max(rectangle(2), 1);
y_end = min(rectangle(2)+rectangle(4), size(oriImg,1));
center = [(x_start + x_end)/2, (y_start + y_end)/2];

% determine scale range
middle_range = (y_end - y_start) / size(oriImg,1) * 1.2;
starting_range = middle_range * param.start_scale;
ending_range = middle_range * param.end_scale;

starting_scale = boxsize/(size(oriImg,1)*ending_range);
ending_scale = boxsize/(size(oriImg,1)*starting_range);
multiplier = 2.^(log2(starting_scale):(1/octave):log2(ending_scale));

% data container for each scale and stage
score = cell(nstage, length(multiplier));
pad = cell(1, length(multiplier));
ori_size = cell(1, length(multiplier));

% change outputs to enable visualizing stagewise results
% note this is why we keep out own copy of m-files of caffe wrapper
features = {};
%interest_layers = {'conv5_2_CPM'; 'Mconv7_stage2'; 'Mconv7_stage3';'Mconv7_stage3';'Mconv7_stage5';'Mconv7_stage6';};

for m = 1:length(multiplier)
    scale = multiplier(m);

    imageToTest = imresize(oriImg, scale);
    ori_size{m} = size(imageToTest);
    center_s = center * scale;
    [imageToTest, pad{m}] = padAround(imageToTest, boxsize, center_s, model.padValue); % into boxsize, which is multipler of 4
    
    imageToTest = preprocess(imageToTest, 0.5, param);
     
    feature = get_feature(imageToTest, net, interest_layers);
    features = [features; feature];
end

time = toc;
fprintf('done, elapsed time: %.3f sec\n', time);


function img_out = preprocess(img, mean, param)
    img_out = double(img)/256;  
    img_out = double(img_out) - mean;
    img_out = permute(img_out, [2 1 3]);
    
    img_out = img_out(:,:,[3 2 1]); % BGR for opencv training in caffe !!!!!
    boxsize = param.model(param.modelID).boxsize;
    centerMapCell = produceCenterLabelMap([boxsize boxsize], boxsize/2, boxsize/2, param.model(param.modelID).sigma);
    img_out(:,:,4) = centerMapCell{1};
    
function scores = applyDNN(images, net, nstage, np)
    input_data = {single(images)};
    % do forward pass to get scores
    % scores are now Width x Height x Channels x Num
    net.forward(input_data);
    scores = cell(1, nstage);
    for s = 1:nstage
        string_to_search = sprintf('stage%d', s);
        blob_id_C = strfind(net.blob_names, string_to_search);
        blob_id = find(not(cellfun('isempty', blob_id_C)));
        blob_id = blob_id(end);
        scores{s} = net.blob_vec(blob_id).get_data();
        if(size(scores{s}, 3) ~= np+1)
            string_to_search = 'conv5_2_CPM';
            blob_id_C = strfind(net.blob_names, string_to_search);
            blob_id = find(not(cellfun('isempty', blob_id_C)));
            blob_id = blob_id(end);
            scores{s} = net.blob_vec(blob_id).get_data();
        end
    end

function feature = get_feature(images, net, layername)
    input_data = {single(images)};
    net.forward(input_data);
    feature = {};
    for i = 1 : length(layername)
        feature = [feature; net.blobs(layername{i}).get_data()];
    end

function [img_padded, pad] = padAround(img, boxsize, center, padValue)
    center = round(center);
    h = size(img, 1);
    w = size(img, 2);
    pad(1) = boxsize/2 - center(2); % up
    pad(3) = boxsize/2 - (h-center(2)); % down
    pad(2) = boxsize/2 - center(1); % left
    pad(4) = boxsize/2 - (w-center(1)); % right
    
    pad_up = repmat(img(1,:,:), [pad(1) 1 1])*0 + padValue;
    img_padded = [pad_up; img];
    pad_left = repmat(img_padded(:,1,:), [1 pad(2) 1])*0 + padValue;
    img_padded = [pad_left img_padded];
    pad_down = repmat(img_padded(end,:,:), [pad(3) 1 1])*0 + padValue;
    img_padded = [img_padded; pad_down];
    pad_right = repmat(img_padded(:,end,:), [1 pad(4) 1])*0 + padValue;
    img_padded = [img_padded pad_right];
    
    center = center + [max(0,pad(2)) max(0,pad(1))];

    img_padded = img_padded(center(2)-(boxsize/2-1):center(2)+boxsize/2, center(1)-(boxsize/2-1):center(1)+boxsize/2, :); %cropping if needed

function [x,y] = findMaximum(map)
    [~,i] = max(map(:));
    [x,y] = ind2sub(size(map), i);
    
function score = resizeIntoScaledImg(score, pad)
    np = size(score,3)-1;
    score = permute(score, [2 1 3]);
    if(pad(1) < 0)
        padup = cat(3, zeros(-pad(1), size(score,2), np), ones(-pad(1), size(score,2), 1));
        score = [padup; score]; % pad up
    else
        score(1:pad(1),:,:) = []; % crop up
    end
    
    if(pad(2) < 0)
        padleft = cat(3, zeros(size(score,1), -pad(2), np), ones(size(score,1), -pad(2), 1));
        score = [padleft score]; % pad left
    else
        score(:,1:pad(2),:) = []; % crop left
    end
    
    if(pad(3) < 0)
        paddown = cat(3, zeros(-pad(3), size(score,2), np), ones(-pad(3), size(score,2), 1));
        score = [score; paddown]; % pad down
    else
        score(end-pad(3)+1:end, :, :) = []; % crop down
    end
    
    if(pad(4) < 0)
        padright = cat(3, zeros(size(score,1), -pad(4), np), ones(size(score,1), -pad(4), 1));
        score = [score padright]; % pad right
    else
        score(:,end-pad(4)+1:end, :) = []; % crop right
    end
    score = permute(score, [2 1 3]);
    
function label = produceCenterLabelMap(im_size, x, y, sigma)
    % this function generates a gaussian peak centered at position (x,y)
    % it is only for center map in testing
    [X,Y] = meshgrid(1:im_size(1), 1:im_size(2));
    X = X - x;
    Y = Y - y;
    D2 = X.^2 + Y.^2;
    Exponent = D2 ./ 2.0 ./ sigma ./ sigma;
    label{1} = exp(-Exponent);
