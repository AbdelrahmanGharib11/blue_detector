% Enhanced Face Detection System with Advanced Feature Verification
% Modified version without cascadedobjectdetector dependency

function detect2(image_path)
   % Clear workspace and command window
    
    close all;
    
    % Starting message for logging
    disp('Starting face detection process...');
    
    % Validate input arguments
    if nargin < 1 || isempty(image_path)
        error('Image path is required');
    end
    
    % Validate image file
    try
        img = imread(image_path);
        disp(['Successfully loaded image from: ', image_path]);
    catch e
        error(['Failed to load image: ', e.message]);
    end
    
    % Create figure for processing visualization (hidden)
    h_fig = figure('Visible', 'off');
    
    % STEP 1: Multi-method Face Detection Approach
    % Method 1: Enhanced Skin Detection
    disp('Applying skin detection...');
    skinMask = detectSkinColor(img);
    
    % STEP 2: Improved Morphological Operations
    disp('Applying morphological operations...');
    skinMask = cleanSkinMask(skinMask);
    
    % STEP 3: Enhanced Region Filtering
    disp('Finding face regions...');
    [labeledMask, regions] = findFaceRegions(skinMask);
    
    % STEP 4: Multiple Feature Verification
    disp('Verifying face candidates...');
    faceBoundingBoxes = verifyFaceCandidates(img, regions);
    
    % Method 2: Try Viola-Jones cascade object detector if available
    try
        disp('Attempting Viola-Jones detection...');
        violaJonesBoxes = detectViolaJones(img);
        % Combine results from both methods
        faceBoundingBoxes = combineDetections(faceBoundingBoxes, violaJonesBoxes);
        disp('Combined detection results successfully');
    catch e
        disp(['Viola-Jones detector failed: ', e.message]);
        disp('Using only skin-based detection.');
    end
    
    % If still no faces detected, try with adaptive thresholding approach
    if isempty(faceBoundingBoxes)
        disp('No faces detected with standard approaches, trying adaptive method...');
        faceBoundingBoxes = tryAdaptiveDetection(img, skinMask);
    end
    
    % STEP 5: Draw Results on Original Image and save
    disp('Drawing results and saving output image...');
    output_image = drawResultsOnImage(img, faceBoundingBoxes);
    
    % Generate guaranteed valid output path
    try
        % Get absolute path of input image
        image_path = fullfile(image_path);
     [filepath, name, ext] = fileparts(image_path);
    output_filename = [name, '_output', ext];  % Consistent format
    output_path = fullfile(filepath, output_filename);
    
    % Ensure unique filename if needed
    counter = 1;
    while exist(output_path, 'file')
        output_filename = [name, '_output_', num2str(counter), ext];
        output_path = fullfile(filepath, output_filename);
        counter = counter + 1;
    end
    
    % Save the image and explicitly print the output path
    try
        imwrite(output_image, output_path);
        disp(['OUTPUT_PATH:', output_path]);  % Special format for Flask to parse
    catch e
        error(['Failed to save output image: ', e.message]);
    end
    end
    
    % Provide success message
    disp('Successfully processed image');
    
    % Close all figures
    close all;
end

function output_image = drawResultsOnImage(img, faceBoundingBoxes)
    % Create a copy of the image for drawing
    output_image = img;
    
    % Get image dimensions for text sizing
    [imgHeight, imgWidth, ~] = size(img);
    
    % Determine font size relative to image size
    fontSize = max(ceil(imgWidth/100), 1);
    lineWidth = max(ceil(imgWidth/500), 2);
    
    % Calculate RGB for green text that will be visible
    if mean(img(:)) > 128
        textColor = [0, 180, 0];  % Darker green for bright images
    else
        textColor = [0, 255, 0];  % Bright green for dark images
    end
    
    % If faces were detected
    if ~isempty(faceBoundingBoxes)
        % Draw each bounding box
        for i = 1:size(faceBoundingBoxes, 1)
            % Extract bounding box coordinates
            x = round(faceBoundingBoxes(i,1));
            y = round(faceBoundingBoxes(i,2));
            w = round(faceBoundingBoxes(i,3));
            h = round(faceBoundingBoxes(i,4));
            
            % Ensure coordinates are within image bounds
            x1 = max(1, x);
            y1 = max(1, y);
            x2 = min(imgWidth, x + w);
            y2 = min(imgHeight, y + h);
            
            % Draw rectangle borders
            % Top border
            output_image(y1:y1+lineWidth-1, x1:x2, 1) = textColor(1);
            output_image(y1:y1+lineWidth-1, x1:x2, 2) = textColor(2);
            output_image(y1:y1+lineWidth-1, x1:x2, 3) = textColor(3);
            
            % Bottom border
            output_image(y2-lineWidth+1:y2, x1:x2, 1) = textColor(1);
            output_image(y2-lineWidth+1:y2, x1:x2, 2) = textColor(2);
            output_image(y2-lineWidth+1:y2, x1:x2, 3) = textColor(3);
            
            % Left border
            output_image(y1:y2, x1:x1+lineWidth-1, 1) = textColor(1);
            output_image(y1:y2, x1:x1+lineWidth-1, 2) = textColor(2);
            output_image(y1:y2, x1:x1+lineWidth-1, 3) = textColor(3);
            
            % Right border
            output_image(y1:y2, x2-lineWidth+1:x2, 1) = textColor(1);
            output_image(y1:y2, x2-lineWidth+1:x2, 2) = textColor(2);
            output_image(y1:y2, x2-lineWidth+1:x2, 3) = textColor(3);
            
            % Place face number label
            labelText = sprintf('Face %d', i);
            output_image = insertText(output_image, [x1, max(1, y1-fontSize*1.5)], ...
                           labelText, 'FontSize', fontSize, 'BoxColor', 'black', ...
                           'TextColor', 'green', 'BoxOpacity', 0.6);
        end
        
        % Add detection count at the top of the image
        countText = sprintf('%d faces detected', size(faceBoundingBoxes, 1));
        output_image = insertText(output_image, [10, 10], countText, ...
                      'FontSize', fontSize+2, 'BoxColor', 'black', ...
                      'TextColor', 'green', 'BoxOpacity', 0.6);
    else
        % No faces detected message
        output_image = insertText(output_image, [imgWidth/2-100, imgHeight/2], ...
                      'No faces detected', 'FontSize', fontSize+4, ...
                      'BoxColor', 'black', 'TextColor', 'red', 'BoxOpacity', 0.6);
    end
end

function skinMask = detectSkinColor(img)
    % Enhanced skin color detection using multiple color spaces
    % and adaptive thresholding based on image statistics
    
    % Convert to different color spaces for robust skin detection
    ycbcrImg = rgb2ycbcr(img);
    Cb = ycbcrImg(:,:,2);  % Blue difference
    Cr = ycbcrImg(:,:,3);  % Red difference
    
    hsvImg = rgb2hsv(img);
    H = hsvImg(:,:,1);     % Hue
    S = hsvImg(:,:,2);     % Saturation
    V = hsvImg(:,:,3);     % Value
    
    labImg = rgb2lab(img); % Add LAB color space for better skin detection
    a = labImg(:,:,2);     % Red/green component
    b = labImg(:,:,3);     % Blue/yellow component
    
    % Calculate normalized RGB for illumination invariance
    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));
    sumRGB = R + G + B;
    
    % Avoid division by zero
    sumRGB(sumRGB == 0) = 1;
    
    % Calculate normalized RGB
    r = R ./ sumRGB;
    g = G ./ sumRGB;
    
    % Adaptive thresholding based on image statistics
    % Get histogram statistics to adapt thresholds to the specific image
    meanCr = mean(Cr(:));
    stdCr = std(double(Cr(:)));
    meanCb = mean(Cb(:));
    stdCb = std(double(Cb(:)));
    
    % Adjust YCbCr thresholds based on image statistics
    CrMin = max(125, meanCr - 1.5 * stdCr);
    CrMax = min(180, meanCr + 1.5 * stdCr);
    CbMin = max(75, meanCb - 1.5 * stdCb);
    CbMax = min(130, meanCb + 1.5 * stdCb);
    
    % Create masks using multiple color space rules with adaptive thresholds
    ycbcrMask = (Cb >= CbMin) & (Cb <= CbMax) & (Cr >= CrMin) & (Cr <= CrMax);
    
    % HSV rules - more inclusive for diverse skin tones
    hsvMask = (H >= 0) & (H <= 0.2) & (S >= 0.1) & (S <= 0.8) & (V >= 0.2);
    
    % LAB color space rules for skin detection
    labMask = (a >= 5) & (a <= 35) & (b >= 10) & (b <= 60);
    
    % Normalized RGB rules
    rgbMask = (r > 0.3) & (r < 0.7) & (g > 0.2) & (g < 0.5);
    
    % Combine masks with weighted approach
    % Give more weight to YCbCr and LAB which are often better for skin detection
    skinMask = (ycbcrMask & labMask) | (hsvMask & rgbMask) | (ycbcrMask & rgbMask);
end

function cleanedMask = cleanSkinMask(skinMask)
    % Enhanced morphological operations with adaptive structuring elements
    
    % Get image dimensions to adapt structuring element sizes
    [height, width] = size(skinMask);
    imgDiag = sqrt(height^2 + width^2);
    
    % Adaptive structuring element sizes based on image dimensions
    se1Size = max(3, round(imgDiag / 100));  % For closing
    se2Size = max(2, round(imgDiag / 150));  % For opening
    
    % Remove small isolated areas (noise)
    cleanedMask = bwareaopen(skinMask, max(100, round(height * width / 5000)));
    
    % Create structuring elements for morphological operations
    se1 = strel('disk', se1Size);
    se2 = strel('disk', se2Size);
    
    % Close the mask (dilation followed by erosion)
    cleanedMask = imclose(cleanedMask, se1);
    
    % Open the mask (erosion followed by dilation)
    cleanedMask = imopen(cleanedMask, se2);
    
    % Fill holes
    cleanedMask = imfill(cleanedMask, 'holes');
    
    % Second round of area opening with larger size
    cleanedMask = bwareaopen(cleanedMask, max(200, round(height * width / 3000)));
    
    % Edge-preserving smoothing using median filter
    cleanedMask = medfilt2(cleanedMask, [5 5]);
end

function [labeledMask, regions] = findFaceRegions(skinMask)
    % Enhanced connected component analysis with improved filtering
    
    % Label connected components
    [labeledMask, numRegions] = bwlabel(skinMask);
    
    % Get properties of each region with expanded metrics
    regionProps = regionprops(labeledMask, 'Area', 'BoundingBox', 'Centroid', 'Extent', ...
                            'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Solidity', ...
                            'Perimeter', 'EulerNumber');
    
    % Image size for relative calculations
    [imgHeight, imgWidth] = size(skinMask);
    imgArea = imgHeight * imgWidth;
    imgDiag = sqrt(imgHeight^2 + imgWidth^2);
    
    % Initialize output structure
    regions = [];
    
    % Enhanced filtering with more sophisticated criteria
    for i = 1:numRegions
        area = regionProps(i).Area;
        bbox = regionProps(i).BoundingBox;
        solidity = regionProps(i).Solidity;
        extent = regionProps(i).Extent;
        perimeter = regionProps(i).Perimeter;
        eulerNumber = regionProps(i).EulerNumber;
        
        % Extract dimensions
        width = bbox(3);
        height = bbox(4);
        
        % Calculate perimeter-to-area ratio (compact shapes have lower values)
        perimeterAreaRatio = perimeter / sqrt(area);
        
        % Skip regions that are too small (adaptive threshold)
        minAreaThreshold = min(0.01, 500 / imgArea);
        if area < (imgArea * minAreaThreshold)
            continue;
        end
        
        % Skip regions that are too large (adaptive threshold)
        maxAreaThreshold = min(0.6, max(0.3, 30000 / imgArea));
        if area > (imgArea * maxAreaThreshold)
            continue;
        end
        
        % Enhanced aspect ratio filtering (more permissive)
        aspectRatio = height / width;
        if aspectRatio < 0.8 || aspectRatio > 2.2
            continue;
        end
        
        % Enhanced solidity filtering (faces are fairly solid)
        if solidity < 0.5  % More permissive
            continue;
        end
        
        % Enhanced extent filtering
        if extent < 0.35  % More permissive
            continue;
        end
        
        % Check perimeter-to-area ratio (compact shapes like faces have lower values)
        if perimeterAreaRatio > 0.2 * imgDiag / sqrt(imgArea)
            continue;
        end
        
        % Add to output with additional metrics
        region.BoundingBox = bbox;
        region.Area = area;
        region.AspectRatio = aspectRatio;
        region.Solidity = solidity;
        region.Extent = extent;
        region.PerimeterAreaRatio = perimeterAreaRatio;
        region.EulerNumber = eulerNumber;
        regions = [regions; region];
    end
    
    % Sort regions by likelihood of being a face (combines multiple metrics)
    if ~isempty(regions)
        scores = calculateRegionScores(regions);
        [~, sortIndices] = sort(scores, 'descend');
        regions = regions(sortIndices);
    end
end

function scores = calculateRegionScores(regions)
    % Calculate composite scores for regions based on face-like characteristics
    
    scores = zeros(length(regions), 1);
    
    for i = 1:length(regions)
        % Ideal values for face regions (empirically determined)
        idealAspectRatio = 1.3;
        idealSolidity = 0.95;
        idealExtent = 0.75;
        
        % Calculate distance from ideal values (normalized)
        aspectRatioScore = 1 - min(1, abs(regions(i).AspectRatio - idealAspectRatio) / 1.0);
        solidityScore = regions(i).Solidity;
        extentScore = regions(i).Extent;
        
        % Euler number (number of objects minus number of holes)
        % Faces typically have some holes (eyes, mouth)
        eulerScore = 1;
        if isfield(regions(i), 'EulerNumber')
            eulerScore = regions(i).EulerNumber <= 0;
            if eulerScore
                eulerScore = 0.8;
            else
                eulerScore = 0.5;
            end
        end
        
        % Weighted combination
        scores(i) = 0.4 * aspectRatioScore + 0.3 * solidityScore + ...
                    0.2 * extentScore + 0.1 * eulerScore;
    end
end

function faceBoundingBoxes = verifyFaceCandidates(img, regions)
    % Enhanced face verification with improved feature detection
    
    faceBoundingBoxes = [];
    
    if isempty(regions)
        return;
    end
    
    % Convert to grayscale
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    
    % Apply histogram equalization for better feature detection
    grayImg = adapthisteq(grayImg);
    
    % For each candidate region
    for i = 1:length(regions)
        bbox = regions(i).BoundingBox;
        
        % Extract coordinates with padding
        x = max(1, round(bbox(1) - bbox(3) * 0.1));
        y = max(1, round(bbox(2) - bbox(4) * 0.1));
        width = min(size(grayImg, 2) - x + 1, round(bbox(3) * 1.2));
        height = min(size(grayImg, 1) - y + 1, round(bbox(4) * 1.2));
        
        % Ensure box doesn't exceed image boundaries
        x2 = min(size(grayImg, 2), x + width - 1);
        y2 = min(size(grayImg, 1), y + height - 1);
        
        % Extract region with padding
        faceCandidate = grayImg(y:y2, x:x2);
        
        % Skip if region is too small for analysis
        if numel(faceCandidate) < 100
            continue;
        end
        
        % Apply enhanced eye detection
        [hasEyes, eyeScore] = detectEyesInRegion(faceCandidate);
        
        % Apply improved edge analysis
        edgeDensity = calculateEdgeDensity(faceCandidate);
        
        % Apply enhanced symmetry analysis
        symmetryScore = calculateSymmetry(faceCandidate);
        
        % Apply improved texture analysis
        textureScore = calculateTextureVariability(faceCandidate);
        
        % Apply facial feature detection (nose, mouth)
        [hasFeatures, featureScore] = detectFacialFeatures(faceCandidate);
        
        % Calculate gradient coherence (face regions have structured gradients)
        gradientScore = calculateGradientCoherence(faceCandidate);
        
        % Combine scores with optimized weights
        overallScore = 0.25 * eyeScore + 0.2 * symmetryScore + ...
                       0.15 * edgeDensity + 0.15 * textureScore + ...
                       0.15 * featureScore + 0.1 * gradientScore;
        
        % More adaptive threshold based on image quality
        adaptiveThreshold = 0.25;
        
        % Either good eye detection or good overall feature score
        if (hasEyes && eyeScore > 0.4) || overallScore > adaptiveThreshold
            % Adjust bounding box to be more accurate
            adjustedBox = adjustBoundingBox(bbox, hasEyes, eyeScore, size(grayImg));
            faceBoundingBoxes = [faceBoundingBoxes; adjustedBox];
        end
    end
    
    % Apply improved Non-Maximum Suppression
    if size(faceBoundingBoxes, 1) > 1
        faceBoundingBoxes = enhancedNonMaximumSuppression(faceBoundingBoxes);
    end
end

function adjustedBox = adjustBoundingBox(bbox, hasEyes, eyeScore, imgSize)
    % Adjust bounding box based on eye detection and typical face proportions
    
    x = bbox(1);
    y = bbox(2);
    width = bbox(3);
    height = bbox(4);
    
    % If strong eye detection, adjust height to typical face proportions
    if hasEyes && eyeScore > 0.6
        % Eyes are typically at 40-45% from top of face
        eyePosition = 0.42;
        
        % Estimate full face height based on eye position
        newHeight = height / eyePosition;
        
        % Adjust y position to keep eyes at same position
        heightDiff = newHeight - height;
        newY = max(1, y - heightDiff * eyePosition);
        
        % Update box
        y = newY;
        height = min(imgSize(1) - y + 1, newHeight);
    end
    
    % Enforce typical face aspect ratio
    idealAspectRatio = 1.3;  % height/width
    currentAspectRatio = height / width;
    
    if abs(currentAspectRatio - idealAspectRatio) > 0.3
        if currentAspectRatio > idealAspectRatio
            % Too tall, increase width
            newWidth = height / idealAspectRatio;
            widthDiff = newWidth - width;
            x = max(1, x - widthDiff / 2);
            width = min(imgSize(2) - x + 1, newWidth);
        else
            % Too wide, increase height
            newHeight = width * idealAspectRatio;
            heightDiff = newHeight - height;
            y = max(1, y - heightDiff / 2);
            height = min(imgSize(1) - y + 1, newHeight);
        end
    end
    
    adjustedBox = [x, y, width, height];
end

function [hasEyes, score] = detectEyesInRegion(faceRegion)
    % Enhanced eye detection using multiple approaches
    
    % Check if region is valid
    if numel(faceRegion) == 0 || min(size(faceRegion)) < 10
        hasEyes = false;
        score = 0;
        return;
    end
    
    % Resize for consistent analysis
    faceRegion = imresize(faceRegion, [100, 80]);
    
    % Expected eye region (upper part of face)
    eyeRegion = faceRegion(10:45, :);
    
    % APPROACH 1: Gradient-based eye detection
    % Calculate horizontal and vertical gradients
    [Gx, Gy] = imgradientxy(eyeRegion);
    absGx = abs(Gx);
    
    % Enhance gradient map
    enhancedGx = imadjust(mat2gray(absGx));
    
    % Threshold to get potential eye edges
    eyeEdges = enhancedGx > 0.3;
    eyeEdges = bwareaopen(eyeEdges, 5);
    
    % Find connected components in eye edges
    [labeledEyes, numEyeRegions] = bwlabel(eyeEdges);
    eyeProps = regionprops(labeledEyes, 'Area', 'BoundingBox', 'Centroid', 'Extent');
    
    % APPROACH 2: Local minima detection (eyes are often darker)
    % Apply minimum filter to detect dark regions
    darkRegions = imerode(eyeRegion, strel('disk', 2));
    darkRegions = imregionalmin(darkRegions);
    darkRegions = bwareaopen(darkRegions, 3);
    
    % Find connected components in dark regions
    [labeledDark, numDarkRegions] = bwlabel(darkRegions);
    darkProps = regionprops(labeledDark, 'Area', 'BoundingBox', 'Centroid');
    
    % APPROACH 3: Haar-like feature approximation
    % Create simple Haar-like features for eye detection
    haarScore = calculateHaarLikeEyeFeatures(eyeRegion);
    
    % Initialize values
    hasEyes = false;
    score = 0.3;  % Default minimal score
    
    % Check gradient-based detection
    if numEyeRegions >= 2
        centers = zeros(min(numEyeRegions, 10), 2);
        
        for j = 1:min(numEyeRegions, 10)
            if j <= length(eyeProps)
                bbox = eyeProps(j).BoundingBox;
                centers(j,:) = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];
            end
        end
        
        % Check for horizontally aligned components
        centersX = centers(:,1);
        centersY = centers(:,2);
        
        % Find pairs of potential eyes
        for j = 1:size(centers, 1)
            for k = j+1:size(centers, 1)
                % Check horizontal alignment
                yDiff = abs(centersY(j) - centersY(k));
                xDiff = abs(centersX(j) - centersX(k));
                
                % Horizontally separated and vertically aligned
                if yDiff < 5 && xDiff > 10 && xDiff < 50
                    hasEyes = true;
                    score = max(score, 0.6 - (yDiff / 10));
                end
            end
        end
    end
    
    % Check dark region detection
    if numDarkRegions >= 2
        darkCenters = zeros(min(numDarkRegions, 10), 2);
        
        for j = 1:min(numDarkRegions, 10)
            if j <= length(darkProps)
                darkCenters(j,:) = darkProps(j).Centroid;
            end
        end
        
        % Check for horizontally aligned dark regions
        for j = 1:size(darkCenters, 1)
            for k = j+1:size(darkCenters, 1)
                % Check horizontal alignment
                yDiff = abs(darkCenters(j,2) - darkCenters(k,2));
                xDiff = abs(darkCenters(j,1) - darkCenters(k,1));
                
                % Horizontally separated and vertically aligned
                if yDiff < 5 && xDiff > 10 && xDiff < 50
                    hasEyes = true;
                    darkScore = 0.5 - (yDiff / 10);
                    score = max(score, darkScore);
                end
            end
        end
    end
    
    % Incorporate Haar-like feature score
    score = max(score, haarScore);
    
    % If any approach found eyes
    hasEyes = score > 0.4;
end

function haarScore = calculateHaarLikeEyeFeatures(eyeRegion)
    % Simple implementation of Haar-like features for eye detection
    
    [height, width] = size(eyeRegion);
    haarScore = 0;
    
    % Skip if region is too small
    if height < 10 || width < 20
        return;
    end
    
    % Define regions for Haar-like features
    leftEyeX = round(width * 0.25);
    rightEyeX = round(width * 0.75);
    eyeY = round(height * 0.5);
    windowSize = round(min(width, height) * 0.2);
    
    % Extract potential eye regions
    leftEyeRegion = eyeRegion(max(1, eyeY-windowSize):min(height, eyeY+windowSize), ...
                             max(1, leftEyeX-windowSize):min(width, leftEyeX+windowSize));
    rightEyeRegion = eyeRegion(max(1, eyeY-windowSize):min(height, eyeY+windowSize), ...
                              max(1, rightEyeX-windowSize):min(width, rightEyeX+windowSize));
    
    % Simple Haar-like feature: darker center, lighter surroundings
    if numel(leftEyeRegion) > 0 && numel(rightEyeRegion) > 0
        % Calculate intensity differences
        leftCenter = mean(leftEyeRegion(:));
        rightCenter = mean(rightEyeRegion(:));
        
        % Get surrounding regions
        surroundingLeft = eyeRegion(max(1, eyeY-windowSize):min(height, eyeY+windowSize), ...
                                   max(1, leftEyeX-2*windowSize):max(1, leftEyeX-windowSize));
        surroundingRight = eyeRegion(max(1, eyeY-windowSize):min(height, eyeY+windowSize), ...
                                    min(width, rightEyeX+windowSize):min(width, rightEyeX+2*windowSize));
        
        if numel(surroundingLeft) > 0 && numel(surroundingRight) > 0
            surroundLeft = mean(surroundingLeft(:));
            surroundRight = mean(surroundingRight(:));
            
            % Eyes are typically darker than surroundings
            leftDiff = surroundLeft - leftCenter;
            rightDiff = surroundRight - rightCenter;
            
            if leftDiff > 10 && rightDiff > 10
                haarScore = 0.5;
            elseif leftDiff > 10 || rightDiff > 10
                haarScore = 0.3;
            end
        end
    end
end

function [hasFeatures, score] = detectFacialFeatures(faceRegion)
    % Detect facial features like nose and mouth
    
    % Default values
    hasFeatures = false;
    score = 0.3;
    
    % Check if region is valid
    if numel(faceRegion) == 0 || min(size(faceRegion)) < 20
        return;
    end
    
    % Resize for consistent analysis
    faceRegion = imresize(faceRegion, [100, 80]);
    
    % Expected nose region (middle part of face)
    noseRegion = faceRegion(40:60, 30:50);
    
    % Expected mouth region (lower part of face)
    mouthRegion = faceRegion(65:85, 20:60);
    
    % Edge detection for facial features
    noseEdges = edge(noseRegion, 'canny');
    mouthEdges = edge(mouthRegion, 'canny');
    
    % Feature detection based on edge density and patterns
    noseEdgeDensity = sum(noseEdges(:)) / numel(noseEdges);
    mouthEdgeDensity = sum(mouthEdges(:)) / numel(mouthEdges);
    
    % Features detected if edge density is in reasonable range
    hasNose = noseEdgeDensity > 0.05 && noseEdgeDensity < 0.3;
    hasMouth = mouthEdgeDensity > 0.1 && mouthEdgeDensity < 0.4;
    
    % Calculate horizontal edge dominance in mouth region (lips)
    [Gx, Gy] = imgradientxy(mouthRegion);
    horizontalDominance = sum(abs(Gx(:))) / (sum(abs(Gy(:))) + eps);
    
    % Mouths typically have stronger horizontal edges
    hasMouthShape = horizontalDominance > 1.2;
    
    % Combined feature detection
    if (hasNose && hasMouth) || (hasMouth && hasMouthShape)
        hasFeatures = true;
        score = 0.5;
        
        % Bonus for strong mouth shape
        if hasMouthShape
            score = score + 0.2;
        end
    elseif hasNose || hasMouth
        hasFeatures = true;
        score = 0.4;
    end
    
    score = min(0.8, score);  % Cap score
end

function gradientScore = calculateGradientCoherence(faceRegion)
    % Calculate gradient coherence to detect structured patterns in faces
    
    % Resize for consistent analysis
    faceRegion = imresize(faceRegion, [100, 80]);
    
    % Calculate gradients
    [Gx, Gy] = imgradientxy(faceRegion);
    [Gmag, Gdir] = imgradient(Gx, Gy);
    
    % Normalize magnitude
    normGmag = Gmag / max(Gmag(:) + eps);
    
    % Calculate gradient coherence using direction consistency
    % in local neighborhoods
    coherence = zeros(size(Gdir));
    windowSize = 5;
    
    for i = 1+windowSize:size(Gdir,1)-windowSize
        for j = 1+windowSize:size(Gdir,2)-windowSize
            window = Gdir(i-windowSize:i+windowSize, j-windowSize:j+windowSize);
            magWindow = normGmag(i-windowSize:i+windowSize, j-windowSize:j+windowSize);
            
            % Consider only significant gradients
            strongGradients = magWindow > 0.2;
            if sum(strongGradients(:)) > 0
                window = window(strongGradients);
                
                % Calculate direction variance
                dirVariance = circ_var(deg2rad(window(:)));
                
                % Lower variance means higher coherence
                coherence(i,j) = 1 - min(1, dirVariance);
            end
        end
    end
    
    % Average coherence for significant gradients
    strongGradients = normGmag > 0.2;
    if sum(strongGradients(:)) > 0
        gradientScore = mean(coherence(strongGradients));
    else
        gradientScore = 0.3;  % Default score
    end
end

function vr = circ_var(alpha)
    % Simple circular variance calculation for angles
    % Input angles in radians
    
    % Calculate mean resultant length
    r = sqrt(sum(cos(alpha))^2 + sum(sin(alpha))^2) / numel(alpha);
    
    % Circular variance
    vr = 1 - r;
end

function edgeDensity = calculateEdgeDensity(faceRegion)
    % Enhanced edge density calculation with pattern analysis
    
    % Resize for consistent analysis
    faceRegion = imresize(faceRegion, [100, 80]);
    
    % Apply Canny edge detection with optimized parameters
    edges = edge(faceRegion, 'canny', [0.1 0.2]);
    
    % Calculate basic edge density
    basicDensity = sum(edges(:)) / numel(edges);
    
    % Divide face into regions to analyze edge distribution
    % Faces have characteristic edge patterns in different regions
    upperRegion = edges(1:30, :);
    middleRegion = edges(31:60, :);
    lowerRegion = edges(61:end, :);
    
    upperDensity = sum(upperRegion(:)) / numel(upperRegion);
    middleDensity = sum(middleRegion(:)) / numel(middleRegion);
    lowerDensity = sum(lowerRegion(:)) / numel(lowerRegion);
    
    % Calculate density ratios (characteristic for faces)
    upperMiddleRatio = (upperDensity + 0.001) / (middleDensity + 0.001);
    lowerMiddleRatio = (lowerDensity + 0.001) / (middleDensity + 0.001);
    
    % Score based on typical face edge patterns
    patternScore = 0;
    
    % Eyes and eyebrows typically create more edges in upper region
    if upperDensity > middleDensity && upperMiddleRatio > 1 && upperMiddleRatio < 2.5
        patternScore = patternScore + 0.3;
    end
    
    % Mouth typically creates edges in lower region
    if lowerDensity > 0.05 && lowerMiddleRatio > 0.8 && lowerMiddleRatio < 2.0
        patternScore = patternScore + 0.2;
    end
    
    % Overall density should be in a reasonable range for faces
    if basicDensity > 0.05 && basicDensity < 0.2
        patternScore = patternScore + 0.3;
    end
    
    % Normalize pattern score
    patternScore = min(1.0, patternScore);
    
    % Combine basic density with pattern score (weighted combination)
    edgeDensity = 0.4 * (1 - abs(basicDensity - 0.12)/0.07) + 0.6 * patternScore;
end


function displayResults(img, faceBoundingBoxes)
    % Display final results with bounding boxes
    
    figure;
    imshow(img);
    title('Final Face Detection Results');
    hold on;
    
    if ~isempty(faceBoundingBoxes)
        % Draw each bounding box
        for i = 1:size(faceBoundingBoxes, 1)
            rectangle('Position', faceBoundingBoxes(i,:), ...
                     'EdgeColor', 'g', ...
                     'LineWidth', 2);
            
            % Add text label
            text(faceBoundingBoxes(i,1), faceBoundingBoxes(i,2)-10, ...
                 sprintf('Face %d', i), ...
                 'Color', 'g', 'FontWeight', 'bold');
        end
        
        % Display detection count
        text(10, 20, sprintf('%d faces detected', size(faceBoundingBoxes,1)), ...
             'Color', 'g', 'FontWeight', 'bold', 'FontSize', 12, ...
             'BackgroundColor', 'k');
    else
        text(size(img,2)/2-100, size(img,1)/2, 'No faces detected', ...
             'Color', 'r', 'FontWeight', 'bold', 'FontSize', 14, ...
             'BackgroundColor', 'k');
    end
    
    hold off;
end

function finalBoxes = enhancedNonMaximumSuppression(boxes)
    % Improved non-maximum suppression with better overlap handling
    
    % If less than two boxes, no need for NMS
    if size(boxes, 1) < 2
        finalBoxes = boxes;
        return;
    end
    
    % Convert from [x, y, width, height] to [x1, y1, x2, y2]
    boxesForNMS = zeros(size(boxes));
    boxesForNMS(:, 1) = boxes(:, 1);             % x1
    boxesForNMS(:, 2) = boxes(:, 2);             % y1
    boxesForNMS(:, 3) = boxes(:, 1) + boxes(:, 3); % x2
    boxesForNMS(:, 4) = boxes(:, 2) + boxes(:, 4); % y2
    
    % Calculate areas
    areas = (boxesForNMS(:, 3) - boxesForNMS(:, 1) + 1) .* ...
            (boxesForNMS(:, 4) - boxesForNMS(:, 2) + 1);
    
    % Sort boxes by area (largest first)
    [~, sortedIndices] = sort(areas, 'descend');
    boxesForNMS = boxesForNMS(sortedIndices, :);
    
    % Initialize indices to keep
    keepIndices = true(size(boxesForNMS, 1), 1);
    
    % Enhanced NMS algorithm
    for i = 1:size(boxesForNMS, 1) - 1
        if keepIndices(i)
            % Calculate overlap with remaining boxes
            xx1 = max(boxesForNMS(i, 1), boxesForNMS(i+1:end, 1));
            yy1 = max(boxesForNMS(i, 2), boxesForNMS(i+1:end, 2));
            xx2 = min(boxesForNMS(i, 3), boxesForNMS(i+1:end, 3));
            yy2 = min(boxesForNMS(i, 4), boxesForNMS(i+1:end, 4));
            
            % Intersection area
            w = max(0, xx2 - xx1 + 1);
            h = max(0, yy2 - yy1 + 1);
            intersection = w .* h;
            
            % Union area
            union = areas(i) + areas(i+1:end) - intersection;
            
            % IoU
            iou = intersection ./ union;
            
            % Suppress boxes with high overlap
            overlapping = iou > 0.4;  % More lenient threshold than standard 0.5
            
            % Only suppress if the overlapping box is significantly smaller
            suppress = overlapping & (areas(i+1:end) < areas(i) * 0.8);
            keepIndices(i+1:end) = keepIndices(i+1:end) & ~suppress;
        end
    end
    
    % Convert back to original format [x, y, width, height]
    finalBoxesNMS = boxesForNMS(keepIndices, :);
    finalBoxes = zeros(size(finalBoxesNMS));
    finalBoxes(:, 1) = finalBoxesNMS(:, 1);                    % x
    finalBoxes(:, 2) = finalBoxesNMS(:, 2);                    % y
    finalBoxes(:, 3) = finalBoxesNMS(:, 3) - finalBoxesNMS(:, 1); % width
    finalBoxes(:, 4) = finalBoxesNMS(:, 4) - finalBoxesNMS(:, 2); % height
end

function faceBoundingBoxes = tryAdaptiveDetection(img, skinMask)
    % Fallback detection method using adaptive thresholds
    
    faceBoundingBoxes = [];
    
    % Try using the largest skin regions
    stats = regionprops(skinMask, 'BoundingBox', 'Area');
    if ~isempty(stats)
        areas = [stats.Area];
        [~, idx] = sort(areas, 'descend');
        
        % Consider top 5 regions with relaxed size constraints
        for i = 1:min(5, length(idx))
            bbox = stats(idx(i)).BoundingBox;
            area = areas(idx(i));
            
            % Very lenient size constraints
            if area > 500 && area < numel(skinMask)*0.9
                faceBoundingBoxes = [faceBoundingBoxes; bbox];
            end
        end
    end
    
    % Apply size-based filtering if we got too many candidates
    if size(faceBoundingBoxes, 1) > 3
        % Keep only top 3 largest
        bboxAreas = faceBoundingBoxes(:,3) .* faceBoundingBoxes(:,4);
        [~, sortIdx] = sort(bboxAreas, 'descend');
        faceBoundingBoxes = faceBoundingBoxes(sortIdx(1:3), :);
    end
end

% Main function remains the same as in your provided code
% All other functions (detectSkinColor, cleanSkinMask, findFaceRegions, etc.)
% remain exactly as you provided them