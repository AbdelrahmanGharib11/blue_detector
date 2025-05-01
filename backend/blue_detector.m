classdef blue_detector < handle
    % blue_detector Detect objects using the Viola-Jones algorithm
    %   This class implements object detection using a cascade of boosted 
    %   classifiers based on the Viola-Jones algorithm.
    
    properties
        % ClassificationModel - Structure that defines the cascade classifier
        ClassificationModel
        
        % MinSize - Minimum object size [height, width] in pixels
        MinSize = [0, 0]
        
        % MaxSize - Maximum object size [height, width] in pixels
        MaxSize = [inf, inf]
        
        % ScaleFactor - Factor by which to increase the scan window on each pass
        ScaleFactor = 1.1
        
        % MergeThreshold - Threshold for merging overlapping detections
        MergeThreshold = 4
    end
    
    methods
        function obj = blue_detector(varargin)
            % Constructor for blue_detector
            % Usage:
            %   detector = blue_detector() - Creates face detector
            %   detector = blue_detector(modelName) - Creates detector for modelName
            %   detector = blue_detector(__, Name, Value) - Specifies additional parameters
            
            % Default to face detection
            modelName = 'Face';
            
            if nargin > 0 && ischar(varargin{1})
                modelName = varargin{1};
                varargin(1) = [];
            end
            
            % Load the appropriate model
            obj.ClassificationModel = obj.loadModel(modelName);
            
            % Process name-value pairs
            if ~isempty(varargin)
                for i = 1:2:length(varargin)
                    propertyName = varargin{i};
                    propertyValue = varargin{i+1};
                    
                    switch propertyName
                        case 'MinSize'
                            obj.MinSize = propertyValue;
                        case 'MaxSize'
                            obj.MaxSize = propertyValue;
                        case 'ScaleFactor'
                            obj.ScaleFactor = propertyValue;
                        case 'MergeThreshold'
                            obj.MergeThreshold = propertyValue;
                        otherwise
                            error('blue_detector:InvalidProperty', ...
                                'Unknown property: %s', propertyName);
                    end
                end
            end
        end
        
        function bboxes = detect(obj, I)
            % DETECT Detect objects in image
            %   BBOXES = detect(OBJ, I) returns bounding boxes of detected objects
            %   in the input image I. BBOXES is an M-by-4 matrix of [x, y, width, height]
            
            % Input validation
            validateattributes(I, {'numeric'}, {'2d', 'nonsparse', 'real'}, 'detect', 'I');
            
            if size(I, 3) == 3
                % Convert RGB to grayscale
                I = rgb2gray(I);
            end
            
            % Convert to uint8 if not already
            if ~isa(I, 'uint8')
                I = im2uint8(I);
            end
            
            % Create integral image for fast feature computation
            integralImg = cumsum(cumsum(double(I), 1), 2);
            
            % Initialize detection variables
            allBboxes = [];
            
            % Define initial window size
            baseWinSize = obj.ClassificationModel.Size;
            
            % Calculate scales for scanning
            maxScale = min(size(I, 1) / baseWinSize(1), size(I, 2) / baseWinSize(2));
            currentScale = 1;
            
            % Multi-scale detection
            while currentScale <= maxScale
                
                % Calculate current window size
                currentWinSize = round(baseWinSize * currentScale);
                
                % Check if window size is within bounds
                if all(currentWinSize >= obj.MinSize) && all(currentWinSize <= obj.MaxSize)
                    
                    % Calculate step size (can be adjusted for speed/accuracy tradeoff)
                    stepSize = max(1, floor(currentScale));
                    
                    % Scan the image
                    rows = 1:stepSize:size(I, 1) - currentWinSize(1) + 1;
                    cols = 1:stepSize:size(I, 2) - currentWinSize(2) + 1;
                    
                    % For each position
                    for row = rows
                        for col = cols
                            % Extract current window coordinates
                            window = [col, row, currentWinSize(2), currentWinSize(1)];
                            
                            % Run cascade classifier on window
                            if obj.evaluateWindow(integralImg, window)
                                allBboxes = [allBboxes; window];
                            end
                        end
                    end
                end
                
                % Increase scale for next iteration
                currentScale = currentScale * obj.ScaleFactor;
            end
            
            % Merge overlapping detections
            bboxes = obj.mergeDetections(allBboxes);
        end
    end
    
    methods (Access = private)
        function model = loadModel(obj, modelName)
            % LOADMODEL Load a pre-trained cascade model
            % This is a simplified implementation that would normally load
            % XML files containing Haar cascade data
            
            % In a real implementation, this would load XML files
            % Here we create a simplified model structure
            
            switch lower(modelName)
                case 'face'
                    % Simplified face detection model
                    model.Size = [24, 24];  % Base window size
                    model.Stages = createSampleFaceModel();
                case 'upperbody'
                    model.Size = [22, 18];
                    model.Stages = createSampleUpperBodyModel();
                case 'profileface'
                    model.Size = [24, 24];
                    model.Stages = createSampleProfileFaceModel();
                otherwise
                    error('blue_detector:UnsupportedModel', ...
                        'Unsupported model: %s', modelName);
            end
            
            % In a real implementation, the createSampleXxxxModel functions
            % would load actual Haar features and thresholds from files
            
            function stages = createSampleFaceModel()
                % This would normally load from XML files
                % For illustration, create a very simplified structure
                stages = struct('threshold', 0.5, ...
                                'features', struct('type', {'rect'}, ...
                                                  'coordinates', {{[2, 7, 4, 4], [8, 7, 4, 4]}}, ...
                                                  'weights', {{1, -2}}, ...
                                                  'threshold', {0.5}));
                % Add more stages in a real implementation
            end
            
            function stages = createSampleUpperBodyModel()
                % Simplified upper body model
                stages = struct('threshold', 0.5, ...
                                'features', struct('type', {'rect'}, ...
                                                  'coordinates', {{[5, 5, 10, 10], [7, 2, 6, 6]}}, ...
                                                  'weights', {{1, -1}}, ...
                                                  'threshold', {0.6}));
            end
            
            function stages = createSampleProfileFaceModel()
                % Simplified profile face model
                stages = struct('threshold', 0.5, ...
                                'features', struct('type', {'rect'}, ...
                                                  'coordinates', {{[10, 5, 8, 10], [5, 5, 5, 10]}}, ...
                                                  'weights', {{1, -1}}, ...
                                                  'threshold', {0.7}));
            end
        end
        
        function isObject = evaluateWindow(obj, integralImg, window)
            % EVALUATEWINDOW Evaluate window using cascade classifier
            % window = [x, y, width, height]
            
            % Extract window coordinates
            x = window(1);
            y = window(2);
            width = window(3);
            height = window(4);
            
            % Scale factor to normalize features to model window size
            scaleX = width / obj.ClassificationModel.Size(2);
            scaleY = height / obj.ClassificationModel.Size(1);
            
            % Compute feature responses
            % For each stage in the cascade
            for stageIdx = 1:length(obj.ClassificationModel.Stages)
                stage = obj.ClassificationModel.Stages(stageIdx);
                
                % Stage sum starts at 0
                stageSum = 0;
                
                % For each feature in the stage
                for featureIdx = 1:length(stage.features)
                    feature = stage.features(featureIdx);
                    
                    % Calculate feature value using integral image
                    featureSum = 0;
                    
                    % For each rectangle in the feature
                    for rectIdx = 1:length(feature.coordinates)
                        % Scale rectangle to current window
                        rectX = x + round(feature.coordinates{rectIdx}(1) * scaleX);
                        rectY = y + round(feature.coordinates{rectIdx}(2) * scaleY);
                        rectW = round(feature.coordinates{rectIdx}(3) * scaleX);
                        rectH = round(feature.coordinates{rectIdx}(4) * scaleY);
                        
                        % Compute rectangle sum using integral image
                        A = integralImg(rectY, rectX);
                        B = integralImg(rectY, rectX + rectW);
                        C = integralImg(rectY + rectH, rectX);
                        D = integralImg(rectY + rectH, rectX + rectW);
                        
                        rectSum = D + A - B - C;
                        
                        % Apply weight
                        featureSum = featureSum + rectSum * feature.weights{rectIdx};
                    end
                    
                    % Apply feature threshold
                    if featureSum > feature.threshold
                        stageSum = stageSum + 1;
                    end
                end
                
                % If stage sum doesn't pass threshold, reject window
                if stageSum < stage.threshold
                    isObject = false;
                    return;
                end
            end
            
            % If all stages passed, it's a detection
            isObject = true;
        end
        
        function mergedBoxes = mergeDetections(obj, boxes)
            % MERGEDETECTIONS Merge overlapping detections
            % This uses a simple non-maximum suppression approach
            
            if isempty(boxes)
                mergedBoxes = [];
                return;
            end
            
            % Sort boxes by size (largest first)
            boxAreas = boxes(:, 3) .* boxes(:, 4);
            [~, sortedIndices] = sort(boxAreas, 'descend');
            boxes = boxes(sortedIndices, :);
            
            % Initialize variables
            numBoxes = size(boxes, 1);
            pickedIndices = false(numBoxes, 1);
            
            % For each box
            for i = 1:numBoxes
                if ~pickedIndices(i)
                    % Mark current box as picked
                    pickedIndices(i) = true;
                    
                    % Get coordinates
                    x1 = boxes(i, 1);
                    y1 = boxes(i, 2);
                    x2 = x1 + boxes(i, 3) - 1;
                    y2 = y1 + boxes(i, 4) - 1;
                    
                    % Count overlapping boxes
                    overlapCount = 1;
                    overlapBoxes = boxes(i, :);
                    
                    % Check against all other boxes
                    for j = i+1:numBoxes
                        if ~pickedIndices(j)
                            % Get comparison box coordinates
                            x1j = boxes(j, 1);
                            y1j = boxes(j, 2);
                            x2j = x1j + boxes(j, 3) - 1;
                            y2j = y1j + boxes(j, 4) - 1;
                            
                            % Compute intersection
                            intersectWidth = min(x2, x2j) - max(x1, x1j) + 1;
                            intersectHeight = min(y2, y2j) - max(y1, y1j) + 1;
                            
                            % If there is overlap
                            if intersectWidth > 0 && intersectHeight > 0
                                % Calculate overlap ratio
                                intersectArea = intersectWidth * intersectHeight;
                                unionArea = boxes(i, 3) * boxes(i, 4) + ...
                                           boxes(j, 3) * boxes(j, 4) - intersectArea;
                                overlapRatio = intersectArea / unionArea;
                                
                                % If overlap is significant
                                if overlapRatio > 0.5
                                    % Mark overlapping box
                                    pickedIndices(j) = true;
                                    overlapCount = overlapCount + 1;
                                    overlapBoxes = [overlapBoxes; boxes(j, :)];
                                end
                            end
                        end
                    end
                    
                    % Check if we have enough overlapping detections
                    if overlapCount >= obj.MergeThreshold
                        % Average the overlapping boxes
                        mergedBox = mean(overlapBoxes, 1);
                        boxes(i, :) = round(mergedBox);
                    else
                        % Not enough overlaps, mark as not picked
                        pickedIndices(i) = false;
                    end
                end
            end
            
            % Return merged boxes
            mergedBoxes = boxes(pickedIndices, :);
        end
    end
end


