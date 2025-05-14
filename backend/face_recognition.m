function face_recognition(inputImagePath, outputJsonPath)
% FACE_RECOGNITION - Headless face recognition for Flask API
%   This function performs face recognition on an input image and outputs
%   results in JSON format for Flask API consumption.
%
%   Parameters:
%     inputImagePath - Path to the image to be analyzed
%     outputJsonPath - Path where JSON result should be saved
%
%   The JSON output contains:
%     - matched_person: Name of the matched person
%     - confidence: Recognition confidence (0-100)
%     - comparison_image: Path to the generated comparison image
%     - test_image: Path to the marked test image
%     - matched_db_image: Path to the matched database image

%% Initialize
try
    % Database folder path (adjust as needed)
    databasePath = "C:\Users\SaWa\OneDrive\Pictures\Saved Pictures\whats\results";
    
    % Create a face detector
    faceDetector = vision.CascadeObjectDetector();
    
    % Check if database folder exists
    if ~exist(databasePath, 'dir')
        error('Database folder not found: %s', databasePath);
    end
    
    % Paths for output images
    outputDir = fileparts(outputJsonPath);
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    comparisonImagePath = fullfile(outputDir, ['comparison_', timestamp, '.jpg']);
    testImagePath = fullfile(outputDir, ['test_', timestamp, '.jpg']);
    matchedImagePath = fullfile(outputDir, ['matched_', timestamp, '.jpg']);
    
    %% Load database images
    fprintf('Loading database images from %s...\n', databasePath);
    
    % Get list of all image files in the database folder
    imageFiles = dir(fullfile(databasePath, '*.jpg'));
    imageFiles = [imageFiles; dir(fullfile(databasePath, '*.png'))];
    imageFiles = [imageFiles; dir(fullfile(databasePath, '*.bmp'))];
    
    % Check if any images were found
    if isempty(imageFiles)
        error('No images found in the database folder: %s', databasePath);
    end
    
    % Display number of images found
    fprintf('Found %d images in the database.\n', length(imageFiles));
    
    % Load database images and extract face features
    databaseFeatures = cell(length(imageFiles), 1);
    dbImgPaths = cell(length(imageFiles), 1);
    personNames = cell(length(imageFiles), 1);
    
    for i = 1:length(imageFiles)
        % Load image
        imgPath = fullfile(databasePath, imageFiles(i).name);
        img = imread(imgPath);
        dbImgPaths{i} = imgPath;
        
        % Get person name from filename (remove extension)
        [~, personName, ~] = fileparts(imageFiles(i).name);
        personNames{i} = strrep(personName, '_', ' '); % Replace underscores with spaces
        
        % Convert to grayscale if needed
        if size(img, 3) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end
        
        % Detect face
        bbox = step(faceDetector, grayImg);
        
        % If no face detected, use whole image
        if isempty(bbox)
            fprintf('Warning: No face detected in %s. Using whole image.\n', imageFiles(i).name);
            faceImg = imresize(grayImg, [100, 100]);
        else
            % If multiple faces detected, use the largest one
            if size(bbox, 1) > 1
                [~, idx] = max(bbox(:,3) .* bbox(:,4)); % Find largest face
                bbox = bbox(idx, :);
            end
            
            % Extract face region and resize to standard size
            faceImg = imcrop(grayImg, bbox);
            faceImg = imresize(faceImg, [100, 100]);
        end
        
        % Apply histogram equalization to normalize lighting
        faceImg = histeq(faceImg);
        
        % Extract features using our robust features function
        databaseFeatures{i} = extractRobustFeatures(faceImg);
        
        % Display progress
        fprintf('Processed %d/%d: %s\n', i, length(imageFiles), personNames{i});
    end
    
    fprintf('Database loaded successfully!\n\n');
    
    %% Process the input image
    fprintf('Processing input image: %s\n', inputImagePath);
    
    % Load the test image
    try
        testImg = imread(inputImagePath);
    catch
        error('Failed to load input image: %s', inputImagePath);
    end
    
    % Process and identify
    [testImgOut, matchedImgOut, matchedPersonName, confidence, bestMatch] = ...
        processAndIdentifyHeadless(testImg, databaseFeatures, personNames, faceDetector, dbImgPaths);
    
    %% Create and save the comparison visualization
    % Create a figure for the comparison result (off-screen)
    hFig = figure('Visible', 'off', 'Position', [100, 100, 1000, 500]);
    
    % Test image with detected face
    subplot(1, 2, 1);
    imshow(testImgOut);
    title('Input Image');
    
    % Matched image result
    subplot(1, 2, 2);
    if ~isempty(matchedImgOut)
        imshow(matchedImgOut);
        title(sprintf('Match: %s (%.1f%% confidence)', matchedPersonName, confidence));
    else
        axis off;
        text(0.5, 0.5, 'No match found', 'FontSize', 18, 'HorizontalAlignment', 'center');
        title('No Match Found');
    end
    
    % Save the comparison figure
    saveas(hFig, comparisonImagePath);
    fprintf('Saved comparison image to: %s\n', comparisonImagePath);
    close(hFig);
    
    % Save the test image
    imwrite(testImgOut, testImagePath);
    fprintf('Saved test image to: %s\n', testImagePath);
    
    % Save matched image if available
    if ~isempty(matchedImgOut)
        imwrite(matchedImgOut, matchedImagePath);
        fprintf('Saved matched image to: %s\n', matchedImagePath);
    else
        matchedImagePath = '';
    end
    
    %% Create JSON result
    resultStruct = struct();
    resultStruct.matched_person = matchedPersonName;
    resultStruct.confidence = confidence;
    resultStruct.comparison_image = comparisonImagePath;
    resultStruct.test_image = testImagePath;
    resultStruct.matched_db_image = matchedImagePath;
    
    % Add top matches (if any)
    if ~isempty(matchedPersonName) && ~strcmpi(matchedPersonName, 'Unknown')
        resultStruct.is_match = true;
    else
        resultStruct.is_match = false;
    end
    
    % Write JSON result
    jsonStr = jsonencode(resultStruct, 'PrettyPrint', true);
    fid = fopen(outputJsonPath, 'w');
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
    
    fprintf('Saved JSON result to: %s\n', outputJsonPath);
    fprintf('Face recognition completed successfully!\n');
    
catch e
    % Handle errors
    fprintf('ERROR: %s\n', e.message);
    
    % Create error JSON result
    errorStruct = struct();
    errorStruct.error = true;
    errorStruct.message = e.message;
    
    % Ensure outputJsonPath directory exists
    if ~exist(fileparts(outputJsonPath), 'dir')
        mkdir(fileparts(outputJsonPath));
    end
    
    % Write error JSON
    jsonStr = jsonencode(errorStruct, 'PrettyPrint', true);
    fid = fopen(outputJsonPath, 'w');
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
end
end

%% Function to extract robust features from face image
function features = extractRobustFeatures(faceImg)
    % This function extracts robust features from face images
    % using a combination of techniques:
    % 1. Image pixel intensities (flattened)
    % 2. Local binary patterns for texture
    % 3. Histogram of oriented gradients for shape
    
    % Ensure input is grayscale
    if size(faceImg, 3) > 1
        faceImg = rgb2gray(faceImg);
    end
    
    % 1. Get pixel intensities as base features
    pixelFeatures = double(faceImg(:));
    
    % 2. Extract simple texture features (simplified LBP-like)
    % Compute horizontal and vertical gradients
    [gx, gy] = gradient(double(faceImg));
    gradMag = sqrt(gx.^2 + gy.^2);
    
    % Flatten gradient features
    gradFeatures = gradMag(:);
    
    % 3. Extract histogram features from different regions
    % Divide the image into 4x4 regions
    [h, w] = size(faceImg);
    regH = floor(h/4);
    regW = floor(w/4);
    
    histFeatures = [];
    for i = 0:3
        for j = 0:3
            % Extract region
            rh = min(regH, h - i*regH);
            rw = min(regW, w - j*regW);
            if rh <= 0 || rw <= 0
                continue;
            end
            
            region = faceImg(i*regH+1:i*regH+rh, j*regW+1:j*regW+rw);
            
            % Compute histogram of the region with 8 bins
            [counts, ~] = histcounts(region, 8);
            
            % Normalize histogram
            if sum(counts) > 0
                counts = counts / sum(counts);
            end
            
            % Add to features
            histFeatures = [histFeatures; counts(:)];
        end
    end
    
    % Combine all features
    features = [pixelFeatures; gradFeatures; histFeatures];
    
    % Normalize all features to have unit norm
    if norm(features) > 0
        features = features / norm(features);
    end
end

%% Function to process image and identify person (headless version)
function [testImgOut, matchedImgOut, matchedPersonName, confidence, bestMatch] = processAndIdentifyHeadless(testImg, databaseFeatures, personNames, faceDetector, dbImgPaths)
    % Initialize return values
    testImgOut = testImg;  % Default to original test image
    matchedImgOut = [];    % Empty if no match found
    matchedPersonName = 'Unknown';
    confidence = 0;
    bestMatch = 0;
    
    % Convert to grayscale if needed
    if size(testImg, 3) == 3
        grayImg = rgb2gray(testImg);
    else
        grayImg = testImg;
    end
    
    % Detect face
    bbox = step(faceDetector, grayImg);
    
    % If no face detected
    if isempty(bbox)
        fprintf('No face detected in the image.\n');
        return;
    end
    
    % If multiple faces detected, use the largest one
    if size(bbox, 1) > 1
        [~, idx] = max(bbox(:,3) .* bbox(:,4)); % Find largest face
        bbox = bbox(idx, :);
    end
    
    % Mark the detected face in the image
    testImgMarked = insertShape(testImg, 'Rectangle', bbox, 'LineWidth', 3, 'Color', 'yellow');
    
    % Update the test image output to have the face marked
    testImgOut = testImgMarked;
    
    % Extract face region and resize to standard size
    faceImg = imcrop(grayImg, bbox);
    faceImg = imresize(faceImg, [100, 100]);
    
    % Apply histogram equalization to normalize lighting
    faceImg = histeq(faceImg);
    
    % Extract robust features from the test face
    testFeatures = extractRobustFeatures(faceImg);
    
    % Calculate similarity with each database image
    similarities = zeros(length(databaseFeatures), 1);
    
    for i = 1:length(databaseFeatures)
        % Calculate similarity based on feature dimensionality
        if numel(testFeatures) == numel(databaseFeatures{i})
            % If dimensions match, use direct comparison
            % Normalize both feature vectors
            normTestFeatures = testFeatures / norm(testFeatures);
            normDBFeatures = databaseFeatures{i} / norm(databaseFeatures{i});
            
            % Calculate cosine similarity (dot product of normalized vectors)
            similarities(i) = dot(normTestFeatures, normDBFeatures);
        else
            % If dimensions don not match (fallback), use simpler method
            % Calculate a similarity metric based on feature type
            % For feature vectors, use Euclidean distance
            dist = sqrt(sum((testFeatures(:) - databaseFeatures{i}(:)).^2));
            similarities(i) = 1 / (1 + dist);  % Convert to similarity (higher is better)
        end
        
        % Display similarity value for debugging
        fprintf('Similarity with %s: %.4f\n', personNames{i}, similarities(i));
    end
    
    % Find the best match
    [bestSimilarity, bestMatch] = max(similarities);
    
    % Find top 3 matches for more robust results
    [sortedSimilarities, sortedIndices] = sort(similarities, 'descend');
    topMatches = sortedIndices(1:min(3, length(sortedIndices)));
    topSimilarities = sortedSimilarities(1:min(3, length(sortedIndices)));
    
    % Set recognition threshold
    recognitionThreshold = 0.3; % Lower threshold for more matches
    
    if bestSimilarity > recognitionThreshold
        % Get the best matching person name
        matchedPersonName = personNames{bestMatch};
        confidence = bestSimilarity * 100; % Convert to percentage
        
        % Load the matching image from the database
        try
            matchedImgOut = imread(dbImgPaths{bestMatch});
        catch e
            fprintf('Warning: Could not load matched image: %s\n', e.message);
            % Create a placeholder image
            matchedImgOut = ones(size(testImg)) * 128;
        end
        
        fprintf('Match found: %s (%.1f%% confidence)\n', matchedPersonName, confidence);
        
        % Display top 3 matches in console
        fprintf('Top matches:\n');
        for i = 1:length(topMatches)
            fprintf('%d. %s (%.1f%% confidence)\n', i, personNames{topMatches(i)}, topSimilarities(i)*100);
        end
    else
        fprintf('No match found in the database.\n');
    end
end
