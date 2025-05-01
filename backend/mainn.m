% Enhanced Face Detection System with Advanced Feature Verification
% Complete version with all functions implemented

function mainn(image)
    % Clear workspace and command window
    clc;
    close all;
    
    % Read input image
    img = imread(image);
    figure(1), imshow(img), title('Original Image');
    
    % STEP 1: Enhanced Skin Detection
    skinMask = detectSkinColor(img);
    figure(2), imshow(skinMask), title('Initial Skin Color Mask');
    
    % STEP 2: Improved Morphological Operations
    skinMask = cleanSkinMask(skinMask);
    figure(3), imshow(skinMask), title('Cleaned Skin Mask');
    
    % STEP 3: Enhanced Region Filtering
    [labeledMask, regions] = findFaceRegions(skinMask);
    figure(4), imshow(label2rgb(labeledMask)), title('Connected Components');
    
    % STEP 4: Multiple Feature Verification with enhanced eye detection
    faceBoundingBoxes = verifyFaceCandidates(img, regions);
    
    % If still no faces detected, try with adaptive thresholding approach
    if isempty(faceBoundingBoxes)
        disp('No faces detected with standard approaches, trying adaptive method...');
        faceBoundingBoxes = tryAdaptiveDetection(img, skinMask);
    end
    
    % STEP 5: Draw Results on Original Image
    displayResults(img, faceBoundingBoxes);
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
        overallScore = 0.35 * eyeScore + 0.15 * symmetryScore + ...
              0.15 * edgeDensity + 0.15 * textureScore + ...
              0.1 * featureScore + 0.1 * gradientScore;

        
        % More adaptive threshold based on image quality
        adaptiveThreshold = 0.22;  % Lower than the original 0.25

        
        % Either good eye detection or good overall feature score
        if (hasEyes && eyeScore > 0.35) || overallScore > adaptiveThreshold
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
    % Enhanced eye detection with multiple approaches
    
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
    
    % APPROACH 1: Enhanced contrast for better feature detection
    enhancedRegion = adapthisteq(eyeRegion);
    
    % APPROACH 2: Multi-scale gradient analysis
    [hasEyesGradient, gradientScore] = detectEyesWithGradient(enhancedRegion);
    
    % APPROACH 3: Template correlation for eye patterns
    [hasEyesTemplate, templateScore] = detectEyesWithTemplates(enhancedRegion);
    
    % APPROACH 4: Improved Haar-like feature detection
    [hasEyesHaar, haarScore] = detectEyesWithHaar(enhancedRegion);
    
    % APPROACH 5: Circle/ellipse detection (eyes are often elliptical)
    [hasEyesEllipse, ellipseScore] = detectEyesByShape(enhancedRegion);
    
    % Combine all approaches with weighted scoring
    combinedScore = 0.25 * gradientScore + 0.2 * templateScore + ...
                    0.25 * haarScore + 0.3 * ellipseScore;
    
    % More permissive threshold since we don't have Viola-Jones
    hasEyes = combinedScore > 0.35;
    score = combinedScore;
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
    
    % Combine basic density with pattern score
    edgeDensity = 0.4 * (1 - abs(basicDensity - 0.12) / 0.07) + 0.6 * patternScore;
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

function [hasEyesGradient, gradientScore] = detectEyesWithGradient(eyeRegion)
    % Detect eyes using gradient information
    
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
    
    % Check for horizontally aligned components
    hasEyesGradient = false;
    gradientScore = 0.3;  % Default score
    
    if numEyeRegions >= 2
        centers = zeros(min(numEyeRegions, 10), 2);
        
        for j = 1:min(numEyeRegions, 10)
            if j <= length(eyeProps)
                bbox = eyeProps(j).BoundingBox;
                centers(j,:) = [bbox(1) + bbox(3)/2, bbox(2) + bbox(4)/2];
            end
        end
        
        % Find pairs of potential eyes
        for j = 1:size(centers, 1)
            for k = j+1:size(centers, 1)
                % Check horizontal alignment
                yDiff = abs(centers(j,2) - centers(k,2));
                xDiff = abs(centers(j,1) - centers(k,1));
                
                % Horizontally separated and vertically aligned
                if yDiff < 5 && xDiff > 10 && xDiff < 50
                    hasEyesGradient = true;
                    gradientScore = max(gradientScore, 0.6 - (yDiff / 10));
                end
            end
        end
    end
end

function [hasEyesTemplate, templateScore] = detectEyesWithTemplates(eyeRegion)
    % Use template matching to detect eyes
    
    % Define simplified eye templates
    eyeTemplate1 = fspecial('gaussian', [7, 9], 2);
    eyeTemplate2 = fspecial('gaussian', [5, 11], 2);
    
    % Normalize templates
    eyeTemplate1 = eyeTemplate1 / max(eyeTemplate1(:));
    eyeTemplate2 = eyeTemplate2 / max(eyeTemplate2(:));
    
    % Perform template matching
    response1 = normxcorr2(eyeTemplate1, eyeRegion);
    response2 = normxcorr2(eyeTemplate2, eyeRegion);
    
    % Find correlation peaks
    maxCorr1 = max(response1(:));
    maxCorr2 = max(response2(:));
    
    % Find peak locations
    [y1, x1] = find(response1 == maxCorr1, 1);
    [y2, x2] = find(response2 == maxCorr2, 1);
    
    % Check if peaks are horizontally aligned and separated
    hasEyesTemplate = false;
    templateScore = 0;
    
    if ~isempty(y1) && ~isempty(y2) && ~isempty(x1) && ~isempty(x2)
        if abs(y1 - y2) < 10 && abs(x1 - x2) > 15 && abs(x1 - x2) < 50
            hasEyesTemplate = true;
            templateScore = min(1, (maxCorr1 + maxCorr2) / 1.8);
        else
            % Find top 5 peaks and check if any pair is aligned
            peaks1 = findTopNPeaks(response1, 5);
            peaks2 = findTopNPeaks(response2, 5);
            
            [hasEyesTemplate, pairScore] = checkAlignedPeaks(peaks1, peaks2);
            templateScore = pairScore * min(1, (maxCorr1 + maxCorr2) / 2);
        end
    end
end

function peaks = findTopNPeaks(responseMap, n)
    % Find top N peaks in a response map
    peaks = [];
    tempMap = responseMap;
    
    for i = 1:n
        [maxVal, maxIdx] = max(tempMap(:));
        if maxVal > 0.5  % Only consider significant peaks
            [y, x] = ind2sub(size(tempMap), maxIdx);
            peaks = [peaks; y, x, maxVal];
            
            % Suppress this peak and its neighborhood
            [rows, cols] = size(tempMap);
            suppSize = 5;
            rMin = max(1, y - suppSize);
            rMax = min(rows, y + suppSize);
            cMin = max(1, x - suppSize);
            cMax = min(cols, x + suppSize);
            tempMap(rMin:rMax, cMin:cMax) = 0;
        else
            break;  % No more significant peaks
        end
    end
end

function [hasAligned, score] = checkAlignedPeaks(peaks1, peaks2)
    % Check if any peaks from two sets form horizontally aligned pairs
    
    hasAligned = false;
    score = 0;
    
    if isempty(peaks1) || isempty(peaks2)
        return;
    end
    
    for i = 1:size(peaks1, 1)
        for j = 1:size(peaks2, 1)
            y1 = peaks1(i, 1);
            x1 = peaks1(i, 2);
            y2 = peaks2(j, 1);
            x2 = peaks2(j, 2);
            
            yDiff = abs(y1 - y2);
            xDiff = abs(x1 - x2);
            
            if yDiff < 8 && xDiff > 15 && xDiff < 60
                hasAligned = true;
                pairScore = (1 - yDiff/10) * (peaks1(i,3) + peaks2(j,3))/2;
                score = max(score, pairScore);
            end
        end
    end
end

function [hasEyesHaar, haarScore] = detectEyesWithHaar(eyeRegion)
    % Improved Haar-like features for eye detection
    
    [height, width] = size(eyeRegion);
    hasEyesHaar = false;
    haarScore = 0;
    
    % Skip if region is too small
    if height < 10 || width < 20
        return;
    end
    
    % Create multiple Haar-like feature templates for eye detection
    % Template 1: Dark center with lighter surroundings (circular model)
    template1 = ones(7, 9);
    template1(2:6, 3:7) = -2;
    
    % Template 2: Horizontal bar model (darker eyebrow + eye)
    template2 = ones(9, 11);
    template2(3:7, 2:9) = -1.5;
    
    % Template 3: Edge-like template for eye corners
    template3 = ones(5, 9);
    template3(:, 1:4) = -1.5;
    
    % Normalize templates
    template1 = template1 / sum(abs(template1(:)));
    template2 = template2 / sum(abs(template2(:)));
    template3 = template3 / sum(abs(template3(:)));
    
    % Apply templates across the image
    response1 = conv2(double(eyeRegion), template1, 'valid');
    response2 = conv2(double(eyeRegion), template2, 'valid');
    response3 = conv2(double(eyeRegion), template3, 'valid');
    response3Flipped = conv2(double(eyeRegion), fliplr(template3), 'valid');
    
    % Find maximum responses
    [maxR1, maxR1Idx] = max(response1(:));
    [maxR2, maxR2Idx] = max(response2(:));
    [maxR3, maxR3Idx] = max(response3(:));
    [maxR3F, maxR3FIdx] = max(response3Flipped(:));
    
    % Convert to 2D indices
    [y1, x1] = ind2sub(size(response1), maxR1Idx);
    [y2, x2] = ind2sub(size(response2), maxR2Idx);
    [y3, x3] = ind2sub(size(response3), maxR3Idx);
    [y3F, x3F] = ind2sub(size(response3Flipped), maxR3FIdx);
    
    % Check if the responses form a plausible eye pattern
    % Approach 1: Check if main templates align horizontally
    if abs(y1 - y2) < 8 && abs(x1 - x2) > 10 && abs(x1 - x2) < 50
        hasEyesHaar = true;
        haarScore = 0.5 + (maxR1 + maxR2) / 4;
    end
    
    % Approach 2: Check if edge templates potentially form eye corners
    if abs(y3 - y3F) < 5 && abs(x3 - x3F) > 10 && abs(x3 - x3F) < 40
        hasEyesHaar = true;
        cornerScore = 0.4 + (maxR3 + maxR3F) / 4;
        haarScore = max(haarScore, cornerScore);
    end
    
    haarScore = min(0.9, haarScore);  % Cap the score
end

function [hasEyesEllipse, ellipseScore] = detectEyesByShape(eyeRegion)
    % Detect eyes based on their elliptical shape
    
    % Edge detection
    edges = edge(eyeRegion, 'canny', [0.1, 0.3]);
    
    % Find circles/ellipses using Hough transform if available
    try
        [centers, radii] = imfindcircles(edges, [3, 8], 'Sensitivity', 0.85);
        
        hasEyesEllipse = false;
        ellipseScore = 0.2;  % Default low score
        
        if length(radii) >= 2
            % Check if circles are horizontally aligned
            centers = sortrows(centers, 1); % Sort by x-coordinate
            
            % Calculate distances between centers
            dists = pdist2(centers, centers);
            
            % Find pairs that are:
            % 1. Horizontally aligned (similar y)
            % 2. Properly separated (not too close, not too far)
            found = false;
            maxScore = 0;
            
            for i = 1:size(centers, 1)-1
                for j = i+1:size(centers, 1)
                    yDiff = abs(centers(i,2) - centers(j,2));
                    xDiff = abs(centers(i,1) - centers(j,1));
                    
                    % Similar radius is a good sign of eye pair
                    radiusDiff = abs(radii(i) - radii(j)) / max(radii(i), radii(j));
                    
                    if yDiff < 5 && xDiff > 15 && xDiff < 60 && radiusDiff < 0.3
                        found = true;
                        pairScore = (1 - yDiff/10) * (1 - radiusDiff);
                        maxScore = max(maxScore, pairScore);
                    end
                end
            end
            
            hasEyesEllipse = found;
            ellipseScore = maxScore;
        end
    catch
        % Fallback if Hough transform isn't available
        % Use simpler blob detection
        stats = regionprops(bwareaopen(edges, 5), 'Area', 'Centroid', 'Eccentricity');
        
        hasEyesEllipse = false;
        ellipseScore = 0.2;  % Default low score
        
        if length(stats) >= 2
            % Filter by eccentricity (eyes are somewhat elliptical)
            validBlobs = find([stats.Eccentricity] > 0.5 & [stats.Eccentricity] < 0.9 & ...
                             [stats.Area] > 10);
            
            if length(validBlobs) >= 2
                centers = reshape([stats(validBlobs).Centroid], 2, [])';
                
                % Check for horizontally aligned pairs
                maxScore = 0;
                for i = 1:size(centers, 1)-1
                    for j = i+1:size(centers, 1)
                        yDiff = abs(centers(i,2) - centers(j,2));
                        xDiff = abs(centers(i,1) - centers(j,1));
                        
                        if yDiff < 5 && xDiff > 15 && xDiff < 60
                            hasEyesEllipse = true;
                            pairScore = 0.6 - (yDiff / 10);
                            maxScore = max(maxScore, pairScore);
                        end
                    end
                end
                
                ellipseScore = maxScore;
            end
        end
    end
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

function symmetryScore = calculateSymmetry(faceRegion)
    % Calculate vertical symmetry score of face region
    
    % Resize for consistent analysis
    faceRegion = imresize(faceRegion, [100, 80]);
    
    % Flip left-right for symmetry comparison
    flippedRegion = fliplr(faceRegion);
    
    % Calculate absolute difference
    diff = abs(double(faceRegion) - double(flippedRegion));
    
    % Normalize difference score
    maxDiff = max(diff(:));
    if maxDiff > 0
        diffScore = mean(diff(:)) / maxDiff;
    else
        diffScore = 0;
    end
    
    % Symmetry score is inverse of difference
    symmetryScore = 1 - diffScore;
    
    % Apply non-linear mapping to emphasize good symmetry
    symmetryScore = symmetryScore^2;
end

function textureScore = calculateTextureVariability(faceRegion)
    % Calculate texture variability using local binary patterns
    
    % Resize for consistent analysis
    faceRegion = imresize(faceRegion, [100, 80]);
    
    % Calculate LBP features
    lbpFeatures = extractLBPFeatures(faceRegion);
    
    % Simple texture score based on LBP variance
    textureScore = min(1, var(lbpFeatures) * 10);
end

function lbpFeatures = extractLBPFeatures(region)
    % Simplified Local Binary Pattern feature extraction
    
    % Convert to double
    region = double(region);
    
    % Initialize LBP image
    lbpImage = zeros(size(region));
    
    % Define neighborhood (simplified 3x3)
    [rows, cols] = size(region);
    
    for i = 2:rows-1
        for j = 2:cols-1
            center = region(i,j);
            code = 0;
            
            % Compare with 8 neighbors
            code = code + (region(i-1,j-1) > center) * 1;
            code = code + (region(i-1,j) > center) * 2;
            code = code + (region(i-1,j+1) > center) * 4;
            code = code + (region(i,j+1) > center) * 8;
            code = code + (region(i+1,j+1) > center) * 16;
            code = code + (region(i+1,j) > center) * 32;
            code = code + (region(i+1,j-1) > center) * 64;
            code = code + (region(i,j-1) > center) * 128;
            
            lbpImage(i,j) = code;
        end
    end
    
    % Calculate histogram (simplified feature)
    lbpFeatures = histcounts(lbpImage(2:end-1,2:end-1), 0:256);
end