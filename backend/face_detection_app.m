%% Simple Face Recognition System
% This script implements a direct comparison face recognition system:
% 1. Accesses a folder with team members' images (named after each person)
% 2. Asks for a test image to identify
% 3. Compares the test image against all database images
% 4. Outputs the best match

%% Initialize and check for database folder
clear;
clc;
close all;

% Database folder path
databasePath = "C:\Users\SaWa\OneDrive\Pictures\Saved Pictures\whats\results";

% Check if database folder exists
if ~exist(databasePath, 'dir')
    % Create folder if it doesn't exist
    mkdir(databasePath);
    fprintf('Database folder created at: %s\n', databasePath);
    fprintf('Please add team member images to this folder.\n');
    fprintf('Each image should be named after the person (e.g., "John_Smith.jpg").\n\n');
    
    % Ask user if they want to continue
    userInput = input('Would you like to add images now? Press Enter when ready or type "exit" to quit: ', 's');
    if strcmpi(userInput, 'exit')
        return;
    end
    
    % Allow time for user to add images
    fprintf('Please add your images to the folder now.\n');
    fprintf('Press Enter when you have finished adding images...\n');
    pause;
end

%% Load database images
fprintf('Loading database images...\n');

% Get list of all image files in the database folder
imageFiles = dir(fullfile(databasePath, '*.jpg'));
imageFiles = [imageFiles; dir(fullfile(databasePath, '*.png'))];
imageFiles = [imageFiles; dir(fullfile(databasePath, '*.bmp'))];

% Check if any images were found
if isempty(imageFiles)
    fprintf('No images found in the database folder.\n');
    fprintf('Please add images to: %s\n', databasePath);
    return;
end

% Display number of images found
fprintf('Found %d images in the database.\n', length(imageFiles));

% Load database images and extract face features
databaseFeatures = cell(length(imageFiles), 1);
personNames = cell(length(imageFiles), 1);

% Create face detector
faceDetector = vision.CascadeObjectDetector();

% Create a debugging figure to visualize face extraction
debugFig = figure('Name', 'Face Processing', 'Position', [150, 150, 800, 600]);

for i = 1:length(imageFiles)
    % Load image
    imgPath = fullfile(databasePath, imageFiles(i).name);
    img = imread(imgPath);
    
    % Get person name from filename (remove extension)
    [~, personName, ~] = fileparts(imageFiles(i).name);
    personNames{i} = strrep(personName, '_', ' '); % Replace underscores with spaces
    
    % Display original image
    figure(debugFig);
    subplot(2, 2, 1);
    imshow(img);
    title(['Original: ', personNames{i}]);
    
    % Convert to grayscale if needed
    if size(img, 3) == 3
        grayImg = rgb2gray(img);
    else
        grayImg = img;
    end
    
    % Display grayscale image
    subplot(2, 2, 2);
    imshow(grayImg);
    title('Grayscale');
    
    % Detect face
    bbox = step(faceDetector, grayImg);
    
    % If no face detected, use whole image
    if isempty(bbox)
        fprintf('Warning: No face detected in %s. Using whole image.\n', imageFiles(i).name);
        faceImg = imresize(grayImg, [100, 100]);
        
        % Display the whole image as face
        subplot(2, 2, 3);
        imshow(grayImg);
        title('No Face Detected - Using Whole Image');
    else
        % If multiple faces detected, use the largest one
        if size(bbox, 1) > 1
            [~, idx] = max(bbox(:,3) .* bbox(:,4)); % Find largest face
            bbox = bbox(idx, :);
        end
        
        % Show face detection result
        imgWithRect = insertShape(img, 'Rectangle', bbox, 'LineWidth', 3, 'Color', 'yellow');
        subplot(2, 2, 3);
        imshow(imgWithRect);
        title('Face Detected');
        
        % Extract face region and resize to standard size
        faceImg = imcrop(grayImg, bbox);
        faceImg = imresize(faceImg, [100, 100]);
    end
    
    % Apply histogram equalization to normalize lighting
    faceImg = histeq(faceImg);
    
    % Display the processed face
    subplot(2, 2, 4);
    imshow(faceImg);
    title('Processed Face (Stored in Database)');
    
    % Extract features using our robust features function
    databaseFeatures{i} = extractRobustFeatures(faceImg);
    
    % Display progress
    fprintf('Processed %d/%d: %s\n', i, length(imageFiles), personNames{i});
    pause(1); % Pause to allow viewing
end

% Close the debug figure
if ishandle(debugFig)
    close(debugFig);
end

fprintf('Database loaded successfully!\n\n');

%% Main recognition loop
while true
    % Ask user for test image
    fprintf('Face Recognition System\n');
    fprintf('1. Select image to identify\n');
    fprintf('2. Use webcam\n');
    fprintf('3. Exit\n');
    
    choice = input('Enter your choice (1-3): ');
    
    if choice == 3
        % Exit the program
        fprintf('Exiting program. Goodbye!\n');
        break;
        
    elseif choice == 1
        % Let user select an image
        [file, path] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files'}, 'Select an image');
        
        if file == 0
            % User canceled
            fprintf('Selection canceled.\n\n');
            continue;
        end
        
        % Process selected image
        testImg = imread(fullfile(path, file));
        [testImgOut, matchedImgOut, matchedPersonName, confidence] = processAndIdentify(testImg, databaseFeatures, personNames, faceDetector, databasePath, imageFiles);
        
        % Display the returned images
        fprintf('\nReturned Images:\n');
        fprintf('- Test image size: %dx%dx%d\n', size(testImgOut));
        if ~isempty(matchedImgOut)
            fprintf('- Matched image size: %dx%dx%d\n', size(matchedImgOut));
            fprintf('- Matched person: %s (%.1f%% confidence)\n', matchedPersonName, confidence);
            
            % Create a new figure to display the returned images
            returnedFig = figure('Name', 'Returned Images', 'Position', [300, 300, 800, 400]);
            subplot(1, 2, 1);
            imshow(testImgOut);
            title('Returned Test Image');
            
            subplot(1, 2, 2);
            imshow(matchedImgOut);
            title(sprintf('Returned Match: %s', matchedPersonName));
        else
            fprintf('- No match found\n');
        end
        
    elseif choice == 2
        % Check if webcam is available
        try
            cam = webcam();
            
            % Create figure for display
            hFig = figure('Name', 'Webcam Face Recognition', 'Position', [100, 100, 800, 600]);
            hAx = axes('Parent', hFig);
            
            % Instructions
            fprintf('Webcam activated. Press "c" to capture image or "q" to quit.\n');
            
            % Main webcam loop
            while ishandle(hFig)
                % Capture frame
                img = snapshot(cam);
                
                % Display frame
                image(img, 'Parent', hAx);
                title('Press "c" to capture or "q" to quit');
                axis(hAx, 'image');
                drawnow;
                
                % Check for keypresses
                key = get(hFig, 'CurrentCharacter');
                
                if ~isempty(key)
                    if key == 'c'
                        % Capture current frame for recognition
                        [testImgOut, matchedImgOut, matchedPersonName, confidence] = processAndIdentify(img, databaseFeatures, personNames, faceDetector, databasePath, imageFiles);
                        
                        % Display the returned images in a new figure
                        if ~isempty(matchedImgOut)
                            returnedFig = figure('Name', 'Webcam Match Result', 'Position', [300, 300, 800, 400]);
                            subplot(1, 2, 1);
                            imshow(testImgOut);
                            title('Captured Image');
                            
                            subplot(1, 2, 2);
                            imshow(matchedImgOut);
                            title(sprintf('Match: %s (%.1f%%)', matchedPersonName, confidence));
                            
                            % Let the user explicitly close this window
                            fprintf('Showing match result. Close the window when ready.\n');
                        end
                    elseif key == 'q'
                        % Quit webcam mode
                        break;
                    end
                    
                    % Clear the key
                    set(hFig, 'CurrentCharacter', '');
                end
            end
            
            % Clean up
            clear cam;
            if ishandle(hFig)
                close(hFig);
            end
            
        catch
            fprintf('Error: Webcam not available or not functioning properly.\n');
        end
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

%% Function to process image and identify person
function [testImgOut, matchedImgOut, matchedPersonName, confidence] = processAndIdentify(testImg, databaseFeatures, personNames, faceDetector, databasePath, imageFiles)
    % Create result figure
    resultFig = figure('Name', 'Recognition Result', 'Position', [100, 100, 1000, 500]);
    
    % Initialize return values
    testImgOut = testImg;  % Default to original test image
    matchedImgOut = [];    % Empty if no match found
    matchedPersonName = 'Unknown';
    confidence = 0;
    
    % Display test image
    subplot(1, 2, 1);
    imshow(testImg);
    title('Test Image');
    
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
        subplot(1, 2, 2);
        text(0.5, 0.5, 'No face detected!', 'FontSize', 18, 'HorizontalAlignment', 'center');
        axis off;
        fprintf('No face detected in the image.\n\n');
        return;
    end
    
    % If multiple faces detected, use the largest one
    if size(bbox, 1) > 1
        [~, idx] = max(bbox(:,3) .* bbox(:,4)); % Find largest face
        bbox = bbox(idx, :);
    end
    
    % Mark the detected face in the image
    testImgMarked = insertShape(testImg, 'Rectangle', bbox, 'LineWidth', 3, 'Color', 'yellow');
    subplot(1, 2, 1);
    imshow(testImgMarked);
    title('Detected Face');
    
    % Update the test image output to have the face marked
    testImgOut = testImgMarked;
    
    % Extract face region and resize to standard size
    faceImg = imcrop(grayImg, bbox);
    faceImg = imresize(faceImg, [100, 100]);
    
    % Apply histogram equalization to normalize lighting
    faceImg = histeq(faceImg);
    
    % Extract robust features from the test face
    testFeatures = extractRobustFeatures(faceImg);
    
    % For debug: Display the processed test face
    debugFig = figure('Name', 'Face Comparison Debug', 'Position', [150, 150, 1000, 400]);
    
    % Original test image with face detected
    subplot(1, 4, 1);
    imshow(testImgMarked);
    title('Input Image');
    
    % The extracted and processed face
    subplot(1, 4, 2);
    imshow(faceImg);
    title('Processed Test Face');
    
    % Calculate similarity with each database image
    similarities = zeros(length(databaseFeatures), 1);
    
    % Find the closest match for visualization (even if below threshold)
    bestSimilarity = -1;
    bestMatch = 1;
    
    % Store database image filenames for later retrieval
    dbImgPaths = cell(length(databaseFeatures), 1);
    
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
            % If dimensions don't match (fallback), use simpler method
            % Extract representative image for visualization
            if size(databaseFeatures{i}, 1) > 100*100  % It's likely a feature vector
                dbFaceImg = reshape(1:100*100, [100, 100]); % Placeholder
            else
                % Try to reshape it to an image for visualization
                try
                    dbFaceImg = reshape(databaseFeatures{i}, [100, 100]);
                catch
                    % If reshape fails, create placeholder
                    dbFaceImg = ones(100, 100) * mean(databaseFeatures{i});
                end
            end
            
            % Calculate a similarity metric based on feature type
            % For feature vectors, use Euclidean distance
            dist = sqrt(sum((testFeatures(:) - databaseFeatures{i}(:)).^2));
            similarities(i) = 1 / (1 + dist);  % Convert to similarity (higher is better)
        end
        
        % Store the database image path for later
        dbImgPaths{i} = fullfile(databasePath, imageFiles(i).name);
        
        % For visualization, show the best matching database face
        if i == 1 || similarities(i) > bestSimilarity
            bestSimilarity = similarities(i);
            bestMatch = i;
            
            % Try to show the actual database image
            try
                dbImg = imread(dbImgPaths{bestMatch});
                subplot(1, 4, 3);
                imshow(dbImg);
            catch
                % Fallback if we can't show the actual image
                subplot(1, 4, 3);
                if exist('dbFaceImg', 'var')
                    imshow(uint8(dbFaceImg));
                else
                    text(0.5, 0.5, 'Database Image', 'HorizontalAlignment', 'center');
                    axis off;
                end
            end
            title(['Database: ' personNames{bestMatch}]);
        end
        
        % Display similarity value for debugging
        fprintf('Similarity with %s: %.4f\n', personNames{i}, similarities(i));
    end
    
    % Show similarity distribution
    subplot(1, 4, 4);
    bar(similarities);
    title('Similarity Scores');
    xlabel('Person Index');
    ylabel('Similarity');
    
    % Wait for user to view debug info
    fprintf('Press any key to continue...\n');
    pause;
    
    % Close the debug figure safely
    if ishandle(debugFig)
        close(debugFig);
    end
    
    % Find the best match
    [bestSimilarity, bestMatch] = max(similarities);
    
    % Find top 3 matches for more robust results
    [sortedSimilarities, sortedIndices] = sort(similarities, 'descend');
    topMatches = sortedIndices(1:min(3, length(sortedIndices)));
    topSimilarities = sortedSimilarities(1:min(3, length(sortedIndices)));
    
    % Set recognition threshold (LOWER for more matches but less accuracy)
    recognitionThreshold = 0.3; % Changed from 0.5 to 0.3 to be more lenient
    
    % Display result
    subplot(1, 2, 2);
    
    if bestSimilarity > recognitionThreshold
        % Get the best matching person name
        matchedPersonName = personNames{bestMatch};
        confidence = bestSimilarity * 100; % Convert to percentage
        
        % Load the matching image from the database (using stored path)
        try
            matchedImgOut = imread(dbImgPaths{bestMatch});
            imshow(matchedImgOut);
            title(sprintf('Match: %s (%.1f%% confidence)', matchedPersonName, confidence));
        catch
            text(0.5, 0.5, 'Match Found (Image not available)', 'FontSize', 18, 'HorizontalAlignment', 'center');
            axis off;
            title(sprintf('Match: %s (%.1f%% confidence)', matchedPersonName, confidence));
        end
        
        fprintf('Match found: %s (%.1f%% confidence)\n\n', matchedPersonName, confidence);
        
        % Display top 3 matches in console
        fprintf('Top matches:\n');
        for i = 1:length(topMatches)
            fprintf('%d. %s (%.1f%% confidence)\n', i, personNames{topMatches(i)}, topSimilarities(i)*100);
        end
        fprintf('\n');
    else
        text(0.5, 0.5, 'No match found', 'FontSize', 18, 'HorizontalAlignment', 'center');
        axis off;
        title('No Match Found');
        fprintf('No match found in the database.\n\n');
    end
    
    % Save the result
    resultsPath = 'D:\flutter_projects\blue_detector\backend\recognition_results';
    if ~exist(resultsPath, 'dir')
        mkdir(resultsPath);
    end
    
    % Generate timestamp for filenames
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    
    % Save the comparison figure
    resultFilename = fullfile(resultsPath, ['comparison_', timestamp, '.jpg']);
    saveas(resultFig, resultFilename);
    fprintf('Comparison result saved to: %s\n', resultFilename);
    
    % Additionally, save individual images
    testImgFilename = fullfile(resultsPath, ['test_', timestamp, '.jpg']);
    imwrite(testImgOut, testImgFilename);
    fprintf('Test image saved to: %s\n', testImgFilename);
    
    % Save matched image if available
    if ~isempty(matchedImgOut)
        matchedImgFilename = fullfile(resultsPath, ['matched_', timestamp, '_', strrep(matchedPersonName, ' ', '_'), '.jpg']);
        imwrite(matchedImgOut, matchedImgFilename);
        fprintf('Matched image saved to: %s\n\n', matchedImgFilename);
        
        % Create a structured result file with all information
        resultInfoFilename = fullfile(resultsPath, ['result_info_', timestamp, '.txt']);
        fid = fopen(resultInfoFilename, 'w');
        fprintf(fid, 'Face Recognition Result\n');
        fprintf(fid, '=====================\n');
        fprintf(fid, 'Date & Time: %s\n', datestr(now));
        fprintf(fid, 'Test Image: %s\n', testImgFilename);
        fprintf(fid, 'Matched Person: %s\n', matchedPersonName);
        fprintf(fid, 'Confidence: %.1f%%\n', confidence);
        fprintf(fid, 'Matched Image: %s\n', matchedImgFilename);
        fclose(fid);
        
        fprintf('Detailed result information saved to: %s\n\n', resultInfoFilename);
    end
end