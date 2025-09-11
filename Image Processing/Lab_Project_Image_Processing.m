clear; clc; close all;

%% Load Ground Truth and Input Video
load('export_ground_truth.mat');  % gTruth: MATLAB groundTruth object with bounding boxes
videoFile = '/Applications/MATLAB_R2024b.app/toolbox/images/imdata/traffic.mj2';

%% Initialize Results Table
results = table('Size',[4,4], ...
    'VariableTypes', {'string','double','double','double'}, ...
    'VariableNames', {'Method','Precision','Recall','F1Score'});
results.Method = {'Thresholding'; 'GMM'; 'YOLOv4'; 'Motion Segmentation'};

%% Display Setup
figure; set(gcf, 'Name', 'Detection Methods Comparison');

%% 1. Thresholding Detection
% Uses grayscale + adaptive thresholding + Canny edges + morphological cleanup
[tp, tr, tf] = processMethod(videoFile, gTruth.LabelData.Car, @detectThreshold, 0.3, [15 15], 'Thresholding');
results{1,2:4} = [tp, tr, tf];

%% 2. GMM Detection
% Uses Computer Vision Toolbox's vision.ForegroundDetector (Gaussian Mixture Model)
gmmDetector = vision.ForegroundDetector('NumGaussians', 3, 'NumTrainingFrames', 800, 'LearningRate', 0.01);
[gp, gr, gf] = processMethod(videoFile, gTruth.LabelData.Car, @(f) detectGMM(f, gmmDetector), 0.3, [5 5], 'GMM');
results{2,2:4} = [gp, gr, gf];

%% 3. YOLOv4 Detection
% Uses pretrained yolov4ObjectDetector from Deep Learning Toolbox
yoloDetector = yolov4ObjectDetector('csp-darknet53-coco');
[yp, yr, yf] = processMethod(videoFile, gTruth.LabelData.Car, @(f) detectYOLO(f, yoloDetector), 0.3, [], 'YOLOv4');
results{3,2:4} = [yp, yr, yf];

%% 4. Motion Segmentation
% Custom motion detector with background model and frame differencing
[mp, mr, mf] = processMethod(videoFile, gTruth.LabelData.Car, @detectMotionSegmentation, 0.3, [20 20], 'Motion Segmentation');
results{4,2:4} = [mp, mr, mf];

%% Display Final Results
disp('Benchmarking Results for All Methods:');
disp(results);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: processMethod
% Inputs:
%   videoFile      - Path to video file
%   gtData         - Ground truth cell array from gTruth.LabelData
%   detectionFunc  - Function handle to method-specific detector
%   iouThreshold   - IoU threshold for matching boxes
%   minBBoxSize    - [width height] filter for small detections (optional)
%   methodName     - Name of the method (for display title)
% Outputs:
%   meanPrecision  - Average precision over all frames
%   meanRecall     - Average recall over all frames
%   f1Score        - Combined F1-score
function [meanPrecision, meanRecall, f1Score] = processMethod(videoFile, gtData, detectionFunc, iouThreshold, minBBoxSize, methodName)
    video = VideoReader(videoFile);
    allPrecisions = []; 
    allRecalls = [];
    frameNum = 1;

    while hasFrame(video) && frameNum <= numel(gtData)
        frame = readFrame(video);

        % Get detection bounding boxes from the specified method
        bboxes = detectionFunc(frame);
        if isempty(bboxes), bboxes = zeros(0,4); end

        % Filter out small detections
        if ~isempty(minBBoxSize)
            bboxes = bboxes(bboxes(:,3) >= minBBoxSize(1) & bboxes(:,4) >= minBBoxSize(2), :);
        end

        % Extract ground truth for current frame
        gt = extractGT(gtData{frameNum});
        if isempty(gt), gt = zeros(0,4); end

        % Calculate precision & recall using built-in bboxPrecisionRecall
        if isempty(bboxes) && isempty(gt)
            precision = 1; recall = 1;
        else
            [precision, recall] = bboxPrecisionRecall(bboxes, gt, iouThreshold);
        end

        % Store for final averaging
        allPrecisions(end+1) = precision;
        allRecalls(end+1) = recall;

        % Visualize detections and ground truth
        imshow(frame); hold on;
        for i = 1:size(gt,1)
            rectangle('Position', gt(i,:), 'EdgeColor','green', 'LineWidth',2, 'LineStyle','--');
        end
        for i = 1:size(bboxes,1)
            rectangle('Position', bboxes(i,:), 'EdgeColor','red', 'LineWidth',2);
        end
        title([methodName ' - Frame ' num2str(frameNum)]);
        hold off; pause(1 / video.FrameRate);
        frameNum = frameNum + 1;
    end

    % Final mean scores
    meanPrecision = mean(allPrecisions);
    meanRecall = mean(allRecalls);
    f1Score = 2 * (meanPrecision * meanRecall) / (meanPrecision + meanRecall + eps);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: extractGT
% Converts raw ground truth entry to a numeric [x y w h] matrix.
function gt = extractGT(gtRaw)
    if isstruct(gtRaw)
        gt = cell2mat(arrayfun(@(x) x.Position, gtRaw, 'UniformOutput', false)');
    elseif isempty(gtRaw)
        gt = zeros(0,4);
    else
        gt = gtRaw;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: detectThreshold
% Applies adaptive thresholding, Canny edges, and morphological filtering.
function bboxes = detectThreshold(frame)
    gray = rgb2gray(frame);
    mask = imbinarize(gray, 'adaptive', 'Sensitivity', 0.3);
    edges = edge(gray, 'Canny');
    binary = mask | edges;
    binary = imopen(binary, strel('rectangle',[3 3]));
    binary = imclose(binary, strel('rectangle',[5 5]));
    binary = imdilate(binary, strel('rectangle',[3 3])); % Extra dilation
    stats = regionprops(binary, 'BoundingBox');
    bboxes = vertcat(stats.BoundingBox);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: detectGMM
% Uses MATLAB's vision.ForegroundDetector to extract foreground objects.
% Note: detector must be passed in via anonymous function: @(f) detectGMM(f, detector)
function bboxes = detectGMM(frame, detector)
    mask = detector(rgb2gray(frame));  % Toolbox object call
    mask = imopen(mask, strel('rectangle',[1 1]));
    mask = imclose(mask, strel('rectangle',[3 3]));
    stats = regionprops(mask, 'BoundingBox');
    bboxes = vertcat(stats.BoundingBox);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: detectYOLO
% Uses yolov4ObjectDetector to find objects and filters by label/scores.
% detector must be passed in as @(f) detectYOLO(f, detector)
function bboxes = detectYOLO(frame, detector)
    [bboxes, scores, labels] = detect(detector, frame); % Toolbox function
    valid = scores >= 0.1 & ismember(labels, {'car', 'truck', 'bus', 'motorcycle'});
    bboxes = bboxes(valid, :);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION: detectMotionSegmentation
% Custom background subtraction with running average and dynamic threshold.
function bboxes = detectMotionSegmentation(frame)
    persistent background;
    gray = double(rgb2gray(frame));

    if isempty(background)
        background = gray;
        bboxes = zeros(0,4);
        return;
    end

    % Compute difference and dynamic threshold
    diff = abs(gray - background);
    threshold = mean(diff(:)) + std(diff(:));
    mask = diff > threshold;

    % Morphological cleanup
    mask = imopen(mask, strel('rectangle',[3 3]));
    mask = imclose(mask, strel('rectangle',[7 7]));
    mask = imfill(mask, 'holes');

    % Update background
    alpha = 0.05;
    background = (1 - alpha) * background + alpha * gray;

    % Extract bounding boxes
    stats = regionprops(mask, 'BoundingBox');
    bboxes = vertcat(stats.BoundingBox);
end
