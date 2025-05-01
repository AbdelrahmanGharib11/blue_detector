function faceRecognitionSystem()
    % Create a dark-themed face recognition system
    
    % Set default figure appearance for dark mode
    setDarkTheme();
    
    % Create main figure with dark theme
    mainFig = figure('Name', 'Face Recognition System', ...
                     'NumberTitle', 'off', ...
                     'Color', [0.15 0.15 0.15], ...
                     'Position', [100, 100, 600, 500], ...
                     'MenuBar', 'none', ...
                     'ToolBar', 'none');
    
    % Add logo/banner at the top
    logoPanel = uipanel('Parent', mainFig, ...
                       'BackgroundColor', [0.1 0.1 0.1], ...
                       'Position', [0 0.85 1 0.15]);
    
    % Add title text
    uicontrol('Parent', logoPanel, ...
              'Style', 'text', ...
              'String', 'FACE RECOGNITION SYSTEM', ...
              'FontSize', 15, ...
              'FontWeight', 'bold', ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'BackgroundColor', [0.1 0.1 0.1], ...
              'Position', [150, 25, 300, 40]);
    
    % Create main content panel
    mainPanel = uipanel('Parent', mainFig, ...
                        'BackgroundColor', [0.2 0.2 0.2], ...
                        'Position', [0 0 1 0.85]);
    
    % Create buttons with modern styling
    btnWidth = 200;
    btnHeight = 60;
    btnSpacing = 20;
    startY = 350;
    
    % Register Face button
    uicontrol('Parent', mainPanel, ...
              'Style', 'pushbutton', ...
              'String', 'Register New Face', ...
              'Position', [(600-btnWidth)/2, startY, btnWidth, btnHeight], ...
              'FontSize', 12, ...
              'FontWeight', 'bold', ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'BackgroundColor', [0.2 0.4 0.6], ...
              'Callback', @registerFaceCallback);
          
    % Delete All Faces button
uicontrol('Parent', mainPanel, ...
          'Style', 'pushbutton', ...
          'String', 'Delete All Faces', ...
          'Position', [(600-btnWidth)/2, startY-2*(btnHeight+btnSpacing), btnWidth, btnHeight], ...
          'FontSize', 12, ...
          'FontWeight', 'bold', ...
          'ForegroundColor', [0.9 0.9 0.9], ...
          'BackgroundColor', [0.6 0.4 0.2], ... % Orange-brown color
          'Callback', @deleteAllFacesCallback);
          
    % Identify Face button
    uicontrol('Parent', mainPanel, ...
              'Style', 'pushbutton', ...
              'String', 'Identify Face', ...
              'Position', [(600-btnWidth)/2, startY-btnHeight-btnSpacing, btnWidth, btnHeight], ...
              'FontSize', 12, ...
              'FontWeight', 'bold', ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'BackgroundColor', [0.2 0.6 0.4], ...
              'Callback', @identifyFaceCallback);
    
    % Exit button (move down one position)
uicontrol('Parent', mainPanel, ...
          'Style', 'pushbutton', ...
          'String', 'Exit', ...
          'Position', [(600-btnWidth)/2, startY-3*(btnHeight+btnSpacing), btnWidth, btnHeight], ...
          'FontSize', 12, ...
          'FontWeight', 'bold', ...
          'ForegroundColor', [0.9 0.9 0.9], ...
          'BackgroundColor', [0.6 0.2 0.2], ...
          'Callback', @exitCallback);      
          
    
          
    
    % Status text area
    global statusText;
    statusText = uicontrol('Parent', mainPanel, ...
                          'Style', 'text', ...
                          'String', 'Ready. Please select an option.', ...
                          'Position', [50, 50, 500, 30], ...
                          'FontSize', 10, ...
                          'HorizontalAlignment', 'left', ...
                          'ForegroundColor', [0.8 0.8 0.8], ...
                          'BackgroundColor', [0.2 0.2 0.2]);
    
    % Wait for figure to close
    uiwait(mainFig);
    
    %---- Callback Functions ----
    function registerFaceCallback(~, ~)
        updateStatus('Loading file dialog...');
        registerFace();
    end
    
    function identifyFaceCallback(~, ~)
        updateStatus('Loading file dialog...');
        identifyFace();
    end

function deleteAllFacesCallback(~, ~)
    updateStatus('Preparing to delete all faces...');
    confirmDeleteAllFaces();
end
    
    function exitCallback(~, ~)
        updateStatus('Exiting application...');
        pause(0.5);
        delete(mainFig);
    end
end

function confirmDeleteAllFaces()
    % Create a dark-themed confirmation dialog
    confirmFig = figure('Name', 'Confirm Deletion', ...
                       'Color', [0.15 0.15 0.15], ...
                       'NumberTitle', 'off', ...
                       'MenuBar', 'none', ...
                       'Position', [400, 400, 450, 180], ...
                       'WindowStyle', 'modal');
    
    % Warning icon (yellow triangle)
    axes('Parent', confirmFig, ...
         'Position', [0.05, 0.3, 0.2, 0.5], ...
         'Color', [0.15 0.15 0.15], ...
         'XColor', 'none', ...
         'YColor', 'none');
    
    % Draw warning triangle
    x = [-0.5, 0.5, 0, -0.5];
    y = [-0.4, -0.4, 0.4, -0.4];
    fill(x, y, [0.9 0.8 0.2]); % Yellow triangle
    text(0, 0, '!', 'FontSize', 24, 'FontWeight', 'bold', ...
         'Color', [0.1 0.1 0.1], 'HorizontalAlignment', 'center');
    axis equal;
    
    % Message text
    uicontrol('Parent', confirmFig, ...
              'Style', 'text', ...
              'String', 'Are you sure you want to delete ALL registered faces? This action cannot be undone.', ...
              'Position', [120, 80, 310, 60], ...
              'FontSize', 11, ...
              'HorizontalAlignment', 'left', ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'BackgroundColor', [0.15 0.15 0.15]);
    
    % Yes button (Delete)
    uicontrol('Parent', confirmFig, ...
              'Style', 'pushbutton', ...
              'String', 'DELETE ALL', ...
              'Position', [120, 30, 150, 35], ...
              'FontSize', 11, ...
              'FontWeight', 'bold', ...
              'BackgroundColor', [0.7 0.2 0.2], ... % Brighter red for emphasis
              'ForegroundColor', [0.9 0.9 0.9], ...
              'Callback', @confirmDeleteCallback);
    
    % Cancel button
    uicontrol('Parent', confirmFig, ...
              'Style', 'pushbutton', ...
              'String', 'Cancel', ...
              'Position', [290, 30, 120, 35], ...
              'FontSize', 11, ...
              'FontWeight', 'bold', ...
              'BackgroundColor', [0.3 0.3 0.3], ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'Callback', @cancelDeleteCallback);
    
    function confirmDeleteCallback(~, ~)
        delete(confirmFig);
        deleteAllFaces();
    end
    
    function cancelDeleteCallback(~, ~)
        updateStatus('Deletion cancelled.');
        delete(confirmFig);
    end
end

function deleteAllFaces()
    % Check if the saved_faces directory exists
    if ~exist('saved_faces', 'dir')
        updateStatus('No face database found.');
        errorDialog('No Database', 'No face database was found to delete.');
        return;
    end
    
    % Get list of all files in the directory
    fileList = dir(fullfile('saved_faces', '.'));
    
    % Count how many files will be deleted (excluding . and ..)
    fileCount = 0;
    for i = 1:length(fileList)
        if ~strcmp(fileList(i).name, '.') && ~strcmp(fileList(i).name, '..')
            fileCount = fileCount + 1;
        end
    end
    
    if fileCount == 0
        updateStatus('Face database is already empty.');
        return;
    end
    
    updateStatus('Deleting all registered faces...');
    
    % Try to delete each file
    deleteCount = 0;
    failCount = 0;
    
    for i = 1:length(fileList)
        if ~strcmp(fileList(i).name, '.') && ~strcmp(fileList(i).name, '..')
            filePath = fullfile('saved_faces', fileList(i).name);
            
            % Attempt to delete the file
            try
                delete(filePath);
                deleteCount = deleteCount + 1;
            catch
                failCount = failCount + 1;
            end
        end
    end
    
    % Report results
    if failCount == 0
        updateStatus(['Successfully deleted all registered faces (' num2str(deleteCount) ' files removed).']);
        successDialog('Deletion Complete', ['All registered faces have been deleted successfully. ' ...
                                            num2str(deleteCount) ' files were removed.']);
    else
        updateStatus(['Deleted ' num2str(deleteCount) ' files, but failed to delete ' num2str(failCount) ' files.']);
        errorDialog('Partial Deletion', ['Deleted ' num2str(deleteCount) ' files, but failed to delete ' ...
                                         num2str(failCount) ' files. Check folder permissions.']);
    end
end

function setDarkTheme()
    % Set default colors for all future figures
    set(0, 'DefaultFigureColor', [0.15 0.15 0.15]);
    set(0, 'DefaultAxesColor', [0.2 0.2 0.2]);
    set(0, 'DefaultAxesXColor', [0.8 0.8 0.8]);
    set(0, 'DefaultAxesYColor', [0.8 0.8 0.8]);
    set(0, 'DefaultAxesZColor', [0.8 0.8 0.8]);
    set(0, 'DefaultTextColor', [0.9 0.9 0.9]);
    set(0, 'DefaultUicontrolBackgroundColor', [0.25 0.25 0.25]);
    set(0, 'DefaultUicontrolForegroundColor', [0.9 0.9 0.9]);
    set(0, 'DefaultUipanelBackgroundColor', [0.2 0.2 0.2]);
    set(0, 'DefaultUipanelForegroundColor', [0.9 0.9 0.9]);
end

function updateStatus(message)
    % Update status text
    global statusText;
    if ~isempty(statusText) && ishandle(statusText)
        set(statusText, 'String', message);
        drawnow;
    end
end

function registerFace()
    % Get image path from user with dark-themed dialog
    [filename, pathname] = uigetfile({'.jpg;.png;.bmp', 'Image Files (.jpg, *.png, *.bmp)'}, 'Select image with face to register');
    
    if filename == 0
        updateStatus('Operation cancelled.');
        return;
    end
    
    updateStatus(['Processing image: ' filename]);
    
    % Full path to image
    imagePath = fullfile(pathname, filename);
    
    % Detect face in the image
    [faceImage, originalImage, faceRect] = detectAndExtractFace(imagePath);
    
    if isempty(faceImage)
        updateStatus('No face detected in the image. Registration failed.');
        errorDialog('No Face Detected', 'Could not detect any faces in the selected image.');
        return;
    end
    
    % Create dark-themed figure for face preview
    previewFig = figure('Name', 'Face Preview', ...
                       'Color', [0.15 0.15 0.15], ...
                       'NumberTitle', 'off', ...
                       'MenuBar', 'none', ...
                       'Position', [200, 200, 800, 400]);
    
    % Show the detected face
    subplot(1, 2, 1);
    imshow(originalImage);
    hold on;
    rectangle('Position', faceRect, 'EdgeColor', [0.2 0.8 0.2], 'LineWidth', 2);
    title('Detected Face', 'Color', [0.9 0.9 0.9]);
    
    subplot(1, 2, 2);
    imshow(faceImage);
    title('Extracted Face', 'Color', [0.9 0.9 0.9]);
    
    % Create dark-themed input dialog for name
    nameDialog = figure('Name', 'Enter Name', ...
                        'Color', [0.15 0.15 0.15], ...
                        'NumberTitle', 'off', ...
                        'MenuBar', 'none', ...
                        'Position', [400, 400, 400, 150], ...
                        'WindowStyle', 'modal');
    
    % Name label
    uicontrol('Parent', nameDialog, ...
              'Style', 'text', ...
              'String', 'Enter person name:', ...
              'Position', [50, 100, 300, 20], ...
              'FontSize', 11, ...
              'HorizontalAlignment', 'left', ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'BackgroundColor', [0.15 0.15 0.15]);
    
    % Name input field
    nameField = uicontrol('Parent', nameDialog, ...
                          'Style', 'edit', ...
                          'Position', [50, 70, 300, 25], ...
                          'FontSize', 11, ...
                          'BackgroundColor', [0.25 0.25 0.25], ...
                          'ForegroundColor', [0.9 0.9 0.9]);
    
    % OK button
    uicontrol('Parent', nameDialog, ...
              'Style', 'pushbutton', ...
              'String', 'OK', ...
              'Position', [170, 20, 100, 30], ...
              'FontSize', 11, ...
              'FontWeight', 'bold', ...
              'BackgroundColor', [0.2 0.4 0.6], ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'Callback', @okCallback);
    
    % Cancel button
    uicontrol('Parent', nameDialog, ...
              'Style', 'pushbutton', ...
              'String', 'Cancel', ...
              'Position', [280, 20, 100, 30], ...
              'FontSize', 11, ...
              'FontWeight', 'bold', ...
              'BackgroundColor', [0.6 0.2 0.2], ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'Callback', @cancelCallback);
    
    % Wait for user input
    uiwait(nameDialog);
    
    function okCallback(~, ~)
        personName = get(nameField, 'String');
        
        if isempty(personName)
            updateStatus('Registration cancelled: No name provided.');
            delete(nameDialog);
            delete(previewFig);
            return;
        end
        
        % Continue with registration
        delete(nameDialog);
        
        % Ensure a writable working folder
        homeDir = java.lang.System.getProperty('user.home');  
        % Move MATLAB's current folder to the user's home directory
        cd(char(homeDir));

        % Remove conflicting file
        if exist('saved_faces','file') == 2
            delete('saved_faces');
        end

        % Verify or fix folder write access
        if exist('saved_faces', 'dir')
            [statusAttr, attrValues] = fileattrib('saved_faces');  
            if statusAttr && isfield(attrValues, 'UserWrite') && ~attrValues.UserWrite
                fileattrib('saved_faces', '+w');
            end
        else
            % Create directory if it doesn't exist
            [mkdirStatus, mkdirMsg, mkdirMsgID] = mkdir('saved_faces');
            if ~mkdirStatus
                errorDialog('Directory Error', ['Failed to create ''saved_faces'': ' mkdirMsg]);
                delete(previewFig);
                return;
            end
        end
        
        % Clean the name to create a valid filename
        validName = cleanNameForFilename(personName);
        
        % Save the face with the person's name
        facePath = fullfile('saved_faces', [validName, '.png']);
        imwrite(faceImage, facePath);
        
        % Extract and save features
        updateStatus('Extracting facial features...');
        faceFeatures = extractImprovedFaceFeatures(faceImage);
        featurePath = fullfile('saved_faces', [validName, '_features.mat']);
        save(featurePath, 'faceFeatures');
        
        updateStatus(['Face registered successfully for ' personName]);
        
        % Show success message
        successDialog('Registration Successful', ['Face registered successfully for ' personName]);
        
        delete(previewFig);
    end
    
    function cancelCallback(~, ~)
        updateStatus('Registration cancelled.');
        delete(nameDialog);
        delete(previewFig);
    end
end

function identifyFace()
    % Get image path from user
    [filename, pathname] = uigetfile({'.jpg;.png;.bmp', 'Image Files (.jpg, *.png, *.bmp)'}, 'Select image to identify');
    if filename == 0
        updateStatus('Operation cancelled.');
        return;
    end

    updateStatus(['Processing image: ' filename]);
    
    % Full path to image
    imagePath = fullfile(pathname, filename);

    % Detect face in the image
    [faceImage, originalImage, faceRect] = detectAndExtractFace(imagePath);
    
    if isempty(faceImage)
        updateStatus('No face detected in the image. Identification failed.');
        errorDialog('No Face Detected', 'Could not detect any faces in the selected image.');
        return;
    end

    % Create dark-themed results window
    resultFig = figure('Name', 'Face Identification Results', ...
                      'Color', [0.15 0.15 0.15], ...
                      'NumberTitle', 'off', ...
                      'MenuBar', 'none', ...
                      'Position', [100, 100, 800, 600]);

    % Show the detected face
    subplot(2, 2, 1);
    imshow(originalImage);
    hold on;
    rectangle('Position', faceRect, 'EdgeColor', [0.2 0.8 0.2], 'LineWidth', 2);
    title('Query Image', 'Color', [0.9 0.9 0.9]);

    subplot(2, 2, 2);
    imshow(faceImage);
    title('Query Face', 'Color', [0.9 0.9 0.9]);

    % Extract features from the detected face
    updateStatus('Extracting facial features...');
    queryFeatures = extractImprovedFaceFeatures(faceImage);

    % Check if we have any saved faces
    if ~exist('saved_faces','dir')
        updateStatus('No registered faces found. Please register faces first.');
        errorDialog('No Data', 'No registered faces found. Please register faces first.');
        delete(resultFig);
        return;
    end

    % List all feature files
    featureFiles = dir(fullfile('saved_faces','*_features.mat'));
    if isempty(featureFiles)
        updateStatus('No registered faces found. Please register faces first.');
        errorDialog('No Data', 'No registered faces found. Please register faces first.');
        delete(resultFig);
        return;
    end

    % Preallocate arrays for distances and names
    n = length(featureFiles);
    distances = zeros(1,n);
    names = cell(1,n);

    updateStatus('Comparing with registered faces...');
    
    % Compare against each saved face
    for i = 1:n
        % Load stored features
        data = load(fullfile('saved_faces', featureFiles(i).name));
        storedFeatures = data.faceFeatures;

        % Compute distance
        distances(i) = calculateImprovedFeatureDistance(queryFeatures, storedFeatures);

        % Extract the person name
        [~, fname] = fileparts(featureFiles(i).name);
        names{i} = strrep(fname, '_features', '');
    end

    % Sort distances
    [sortedD, idx] = sort(distances);
    
    % Create text area for results
    resultsTextArea = uicontrol('Parent', resultFig, ...
                               'Style', 'text', ...
                               'Position', [600, 50, 180, 300], ...
                               'FontSize', 10, ...
                               'HorizontalAlignment', 'left', ...
                               'ForegroundColor', [0.9 0.9 0.9], ...
                               'BackgroundColor', [0.2 0.2 0.2]);
    
    % Format results text
    resultsText = 'Match Results:\n\n';
    for k = 1:min(5, n)  % Show top 5 matches
        resultsText = [resultsText, num2str(k), '. ', strrep(names{idx(k)}, '_', ' '), ...
                      ' (', num2str(sortedD(k), '%.4f'), ')\n'];
    end
    
    set(resultsTextArea, 'String', resultsText);

    % Choose a threshold between genuine and impostor scores
    recognitionThreshold = 0.5;  % adjust based on your observed distances

    % Find best match
    [minDistance, bestIdx] = min(distances);
    bestMatch = names{bestIdx};

    % Display result
    if minDistance < recognitionThreshold
        updateStatus(['Face identified as: ', strrep(bestMatch, '_', ' '), ' (distance=', num2str(minDistance, '%.4f'), ')']);
        
        % Show the matching face
        matchFacePath = fullfile('saved_faces', [bestMatch, '.png']);
        if exist(matchFacePath, 'file')
            subplot(2, 2, 3);
            imshow(imread(matchFacePath));
            title(['Matched: ', strrep(bestMatch, '_', ' ')], 'Color', [0.9 0.9 0.9]);
            
            % Show feature comparison visualization
            subplot(2, 2, 4);
            visualizeFeatureComparison(queryFeatures, load(fullfile('saved_faces', [bestMatch, '_features.mat'])).faceFeatures);
            title(sprintf('Feature Similarity: %.1f%%', (1 - minDistance/recognitionThreshold)*100), 'Color', [0.9 0.9 0.9]);
        end
        
        % Add result label
        resultLabel = uicontrol('Parent', resultFig, ...
                               'Style', 'text', ...
                               'String', ['MATCH FOUND: ', strrep(bestMatch, '_', ' ')], ...
                               'Position', [200, 30, 400, 30], ...
                               'FontSize', 14, ...
                               'FontWeight', 'bold', ...
                               'HorizontalAlignment', 'center', ...
                               'ForegroundColor', [0.2 0.8 0.2], ...
                               'BackgroundColor', [0.15 0.15 0.15]);
    else
        updateStatus('No matching face found in the database.');
        
        subplot(2, 2, 3);
        ax = gca;
        ax.Color = [0.2 0.2 0.2];
        text(0.5, 0.5, 'No Match Found', 'FontSize', 14, 'Color', [0.8 0.2 0.2], ...
             'HorizontalAlignment', 'center');
        axis off;
        title('Unknown Person', 'Color', [0.9 0.9 0.9]);
        
        % Add result label
        resultLabel = uicontrol('Parent', resultFig, ...
                               'Style', 'text', ...
                               'String', 'NO MATCH FOUND', ...
                               'Position', [200, 30, 400, 30], ...
                               'FontSize', 14, ...
                               'FontWeight', 'bold', ...
                               'HorizontalAlignment', 'center', ...
                               'ForegroundColor', [0.8 0.2 0.2], ...
                               'BackgroundColor', [0.15 0.15 0.15]);
    end
    
    % Close button
    uicontrol('Parent', resultFig, ...
              'Style', 'pushbutton', ...
              'String', 'Close', ...
              'Position', [350, 10, 100, 30], ...
              'FontSize', 11, ...
              'BackgroundColor', [0.3 0.3 0.3], ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'Callback', @() delete(resultFig));
end

function errorDialog(title, message)
    % Create a dark-themed error dialog
    errFig = figure('Name', title, ...
                   'Color', [0.15 0.15 0.15], ...
                   'NumberTitle', 'off', ...
                   'MenuBar', 'none', ...
                   'Position', [400, 400, 400, 150], ...
                   'WindowStyle', 'modal');
    
    % Error icon (red circle)
    axes('Parent', errFig, ...
         'Position', [0.05, 0.3, 0.2, 0.5], ...
         'Color', [0.15 0.15 0.15], ...
         'XColor', 'none', ...
         'YColor', 'none');
    
    t = linspace(0, 2*pi, 100);
    x = cos(t);
    y = sin(t);
    fill(x, y, [0.8 0.2 0.2]);
    text(0, 0, '!', 'FontSize', 24, 'FontWeight', 'bold', ...
         'Color', [1 1 1], 'HorizontalAlignment', 'center');
    axis equal;
    
    % Message text
    uicontrol('Parent', errFig, ...
              'Style', 'text', ...
              'String', message, ...
              'Position', [120, 60, 260, 60], ...
              'FontSize', 10, ...
              'HorizontalAlignment', 'left', ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'BackgroundColor', [0.15 0.15 0.15]);
    
    % OK button
    uicontrol('Parent', errFig, ...
              'Style', 'pushbutton', ...
              'String', 'OK', ...
              'Position', [150, 20, 100, 30], ...
              'FontSize', 11, ...
              'FontWeight', 'bold', ...
              'BackgroundColor', [0.3 0.3 0.3], ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'Callback', @() delete(errFig));
end

function successDialog(title, message)
    % Create a dark-themed success dialog
    successFig = figure('Name', title, ...
                       'Color', [0.15 0.15 0.15], ...
                       'NumberTitle', 'off', ...
                       'MenuBar', 'none', ...
                       'Position', [400, 400, 400, 150], ...
                       'WindowStyle', 'modal');
    
    % Success icon (green checkmark)
    axes('Parent', successFig, ...
         'Position', [0.05, 0.3, 0.2, 0.5], ...
         'Color', [0.15 0.15 0.15], ...
         'XColor', 'none', ...
         'YColor', 'none');
    
    t = linspace(0, 2*pi, 100);
    x = cos(t);
    y = sin(t);
    fill(x, y, [0.2 0.8 0.2]);
    text(0, 0, '✓', 'FontSize', 24, 'FontWeight', 'bold', ...
         'Color', [1 1 1], 'HorizontalAlignment', 'center');
    axis equal;
    
    % Message text
    uicontrol('Parent', successFig, ...
              'Style', 'text', ...
              'String', message, ...
              'Position', [120, 60, 260, 60], ...
              'FontSize', 10, ...
              'HorizontalAlignment', 'left', ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'BackgroundColor', [0.15 0.15 0.15]);
    
    % OK button
    uicontrol('Parent', successFig, ...
              'Style', 'pushbutton', ...
              'String', 'OK', ...
              'Position', [150, 20, 100, 30], ...
              'FontSize', 11, ...
              'FontWeight', 'bold', ...
              'BackgroundColor', [0.3 0.3 0.3], ...
              'ForegroundColor', [0.9 0.9 0.9], ...
              'Callback', @() delete(successFig));
end

% The following functions remain the same as in the original code
% (but I'll include them for completeness)

function [faceImage, originalImage, faceRect] = detectAndExtractFace(imagePath)
    % Load image
    originalImage = imread(imagePath);
    
    % Convert to grayscale for processing
    if size(originalImage, 3) == 3
        grayImage = rgb2gray(originalImage);
    else
        grayImage = originalImage;
    end
    
    % Step 1: Preprocess image
    % Apply histogram equalization
    equalizedImage = myHistEq(grayImage);
    
    % Apply Gaussian smoothing
    smoothedImage = myGaussianSmooth(equalizedImage, 1.5);
    
    % Initialize faceImage as empty and faceRect
    faceImage = [];
    faceRect = [];
    
    % Step 2: Detect skin regions (if color image)
    if size(originalImage, 3) == 3
        % Detect initial skin regions
        skinMask = detectSkinRegions(originalImage);
        
        % Clean up the mask with morphological operations
        skinMask = imclose(skinMask, strel('disk', 8));
        skinMask = imopen(skinMask, strel('disk', 4));
        
        % Find connected components
        [labeledImage, numRegions] = bwlabel(skinMask);
        regionProps = regionprops(labeledImage, 'BoundingBox', 'Area', 'Eccentricity', 'Extent', 'PixelList');
        
        % Initialize array to store valid face regions
        validFaces = [];
        
        % Step 3: Filter regions based on face-like properties
        for i = 1:numRegions
            % Get region properties
            bbox = regionProps(i).BoundingBox;
            area = regionProps(i).Area;
            eccentricity = regionProps(i).Eccentricity;
            extent = regionProps(i).Extent;
            
            % Calculate aspect ratio of bounding box
            aspectRatio = bbox(3) / bbox(4);
            
            % Filter based on multiple criteria:
            % 1. Size - faces are usually significant portions of the image
            % 2. Shape - faces have specific aspect ratio (width/height)
            % 3. Eccentricity - faces are roughly oval (not too elongated)
            % 4. Extent - ratio of pixels in the region to pixels in the bounding box
            
            minArea = 1000; % Adjust based on your images
            
            if (area > minArea) && ... 
               (aspectRatio > 0.5 && aspectRatio < 1.8) && ... % face aspect ratio
               (eccentricity < 0.85) && ... % not too elongated
               (extent > 0.45) % reasonably filled
                
                % Extract face region for feature detection
                x = max(1, round(bbox(1)));
                y = max(1, round(bbox(2)));
                width = min(size(grayImage, 2) - x, round(bbox(3)));
                height = min(size(grayImage, 1) - y, round(bbox(4)));
                
                faceRegion = grayImage(y:y+height, x:x+width);
                
                % Step 4: Look for face features (eyes)
                if detectEyePair(faceRegion)
                    validFaces = [validFaces, i];
                end
            end
        end
        
        % Step 5: Handle multiple faces with non-maximum suppression
        if ~isempty(validFaces)
            validFaces = nonMaximumSuppression(regionProps, validFaces);
        end
        
        % Extract the best matching face (e.g., largest area)
        if ~isempty(validFaces)
            maxArea = 0;
            bestIndex = validFaces(1);
            for i = 1:length(validFaces)
                bbox = regionProps(validFaces(i)).BoundingBox;
                area = bbox(3) * bbox(4);
                if area > maxArea
                    maxArea = area;
                    bestIndex = validFaces(i);
                end
            end
            
            % Extract the face region from original image
            bbox = regionProps(bestIndex).BoundingBox;
            x = max(1, round(bbox(1)));
            y = max(1, round(bbox(2)));
            width = min(size(originalImage, 2) - x, round(bbox(3)));
            height = min(size(originalImage, 1) - y, round(bbox(4)));
            
            % Store the face rectangle
            faceRect = [x, y, width, height];
            
            % Add some margin around the face (15% on each side)
            marginX = round(width * 0.15);
            marginY = round(height * 0.15);
            
            extX = max(1, x - marginX);
            extY = max(1, y - marginY);
            extWidth = min(size(originalImage, 2) - extX, width + 2*marginX);
            extHeight = min(size(originalImage, 1) - extY, height + 2*marginY);
            
            faceImage = originalImage(extY:extY+extHeight-1, extX:extX+extWidth-1, :);
            
            % Normalize size for consistent recognition (160x160 is good for faces)
            faceImage = imresize(faceImage, [160, 160]);
        end
    end
end

function features = extractImprovedFaceFeatures(faceImage)
    % Convert to grayscale if it's a color image
    if size(faceImage, 3) == 3
        grayFace = rgb2gray(faceImage);
    else
        grayFace = faceImage;
    end
    
    % Ensure consistent size
    grayFace = imresize(grayFace, [160, 160]);
    
    % Apply histogram equalization for contrast enhancement
    grayFace = adapthisteq(grayFace, 'ClipLimit', 0.02);
    
    % ---------- Feature Set 1: LBP (Local Binary Patterns) ----------
    % Divide face into regions
    cellSize = 20; % 8x8 cells (20x20 pixels each)
    numCellsX = floor(size(grayFace, 2) / cellSize);
    numCellsY = floor(size(grayFace, 1) / cellSize);
    
    lbpFeatures = [];
    for y = 1:numCellsY
        for x = 1:numCellsX
            % Extract cell
            % Extract cell
            cellX = (x-1)*cellSize + 1;
            cellY = (y-1)*cellSize + 1;
            cell = grayFace(cellY:cellY+cellSize-1, cellX:cellX+cellSize-1);
            
            % Calculate LBP for this cell
            cellLBP = computeLBP(cell);
            
            % Calculate histogram of LBP values (16 bins)
            [counts, ~] = histcounts(cellLBP(:), 16);
            
            % Normalize histogram
            if sum(counts) > 0
                counts = counts / sum(counts);
            end
            
            % Add to feature vector
            lbpFeatures = [lbpFeatures, counts];
        end
    end
    
    % ---------- Feature Set 2: HOG (Histogram of Oriented Gradients) ----------
    % Calculate gradients
    [gx, gy] = imgradientxy(grayFace);
    gradMag = sqrt(gx.^2 + gy.^2);
    gradDir = atan2d(gy, gx);
    gradDir = mod(gradDir, 360); % Convert to range [0, 360°]
    
    % Parameters
    cellSize = 20; % 8x8 cells 
    numBins = 9; % 9 orientation bins (20° each from 0-180°)
    
    hogFeatures = [];
    for y = 1:numCellsY
        for x = 1:numCellsX
            % Extract cell
            cellX = (x-1)*cellSize + 1;
            cellY = (y-1)*cellSize + 1;
            cellMag = gradMag(cellY:cellY+cellSize-1, cellX:cellX+cellSize-1);
            cellDir = gradDir(cellY:cellY+cellSize-1, cellX:cellX+cellSize-1);
            
            % Compute histogram of oriented gradients
            binSize = 360 / numBins;
            cellHist = zeros(1, numBins);
            
            for i = 1:cellSize
                for j = 1:cellSize
                    bin = floor(cellDir(i, j) / binSize) + 1;
                    if bin > numBins
                        bin = 1; % Wrap around
                    end
                    cellHist(bin) = cellHist(bin) + cellMag(i, j);
                end
            end
            
            % Normalize histogram
            if sum(cellHist) > 0
                cellHist = cellHist / sum(cellHist);
            end
            
            % Add to feature vector
            hogFeatures = [hogFeatures, cellHist];
        end
    end
    
    % ---------- Feature Set 3: Gabor Filter Responses ----------
    % Define Gabor filters at different orientations
    numOrientations = 8;
    gaborFeatures = [];
    
    for i = 1:numOrientations
        theta = (i-1) * pi/numOrientations;
        gaborFilter = gabor(5, theta, 'SpatialFrequencyBandwidth', 1);
        gaborMag = imgaborfilt(grayFace, gaborFilter);
        
        % Downsample and extract features
        gaborDown = imresize(gaborMag, [40, 40]); % 40x40
        
        % Add mean and standard deviation as features
        gaborFeatures = [gaborFeatures, mean(gaborDown(:)), std(gaborDown(:))];
    end
    
    % Combine all features and normalize
    features = [lbpFeatures, hogFeatures, gaborFeatures];
    
    % L2 normalization of the entire feature vector
    features = features / norm(features);
end

function lbp = computeLBP(cell)
    [rows, cols] = size(cell);
    lbp = zeros(rows-2, cols-2);
    
    for i = 2:rows-1
        for j = 2:cols-1
            % Get 3x3 neighborhood
            center = cell(i, j);
            neighborhood = [
                cell(i-1,j-1), cell(i-1,j), cell(i-1,j+1), ...
                cell(i,j-1),                cell(i,j+1), ...
                cell(i+1,j-1), cell(i+1,j), cell(i+1,j+1)
            ];
            
            % Calculate LBP code
            binCode = neighborhood >= center;
            
            % Convert binary code to decimal
            weights = [1, 2, 4, 8, 16, 32, 64, 128];
            lbpCode = sum(binCode .* weights);
            
            lbp(i-1, j-1) = lbpCode;
        end
    end
end

function distance = calculateImprovedFeatureDistance(f1, f2)
    % Split features into their components
    numLBP = 512;  % Adjust based on your actual LBP feature size
    numHOG = 576;  % Adjust based on your actual HOG feature size
    % DCT features are the remainder
    
    lbp1 = f1(1:numLBP);
    lbp2 = f2(1:numLBP);
    
    hog1 = f1(numLBP+1:numLBP+numHOG);
    hog2 = f2(numLBP+1:numLBP+numHOG);
    
    dct1 = f1(numLBP+numHOG+1:end);
    dct2 = f2(numLBP+numHOG+1:end);
    
    % Calculate separate distances for each feature type
    distLBP = chiSquareDistance(lbp1, lbp2);  % Chi-square is good for histograms
    distHOG = norm(hog1 - hog2);              % Euclidean for HOG
    distDCT = cosineDistance(dct1, dct2);     % Cosine for DCT
    
    % Weight the feature distances - prioritize structural features
    % These weights should be optimized based on testing
    weights = [0.3, 0.3, 0.4];  % LBP, HOG, DCT
    distance = weights(1)*distLBP + weights(2)*distHOG + weights(3)*distDCT;
end

function dist = chiSquareDistance(x, y)
    % Chi-square distance for histogram features
    numerator = (x - y).^2;
    denominator = x + y + eps;  % Add eps to avoid division by zero
    dist = 0.5 * sum(numerator ./ denominator);
end

function dist = cosineDistance(x, y)
    % Cosine distance
    dist = 1 - (dot(x, y) / (norm(x) * norm(y) + eps));
end

function visualizeFeatureComparison(features1, features2)
    % Create a visualization of feature similarity suitable for dark mode
    numFeatures = length(features1);
    featureGroups = min(20, numFeatures); % Limit visualization to reasonable size
    
    % Group features for visualization
    groupSize = floor(numFeatures / featureGroups);
    groupedFeatures1 = zeros(1, featureGroups);
    groupedFeatures2 = zeros(1, featureGroups);
    
    for i = 1:featureGroups
        startIdx = (i-1) * groupSize + 1;
        endIdx = min(i * groupSize, numFeatures);
        groupedFeatures1(i) = mean(features1(startIdx:endIdx));
        groupedFeatures2(i) = mean(features2(startIdx:endIdx));
    end
    
    % Normalize for visualization
    maxVal = max([groupedFeatures1, groupedFeatures2]);
    if maxVal > 0
        groupedFeatures1 = groupedFeatures1 / maxVal;
        groupedFeatures2 = groupedFeatures2 / maxVal;
    end
    
    % Create bar chart with dark theme colors
    h = bar(1:featureGroups, [groupedFeatures1; groupedFeatures2]');
    set(h(1), 'FaceColor', [0.2 0.6 0.8]); % Blue for query face
    set(h(2), 'FaceColor', [0.2 0.8 0.2]); % Green for database face
    
    % Set grid and axis properties for dark theme
    grid on;
    ax = gca;
    ax.Color = [0.2 0.2 0.2];
    ax.GridColor = [0.5 0.5 0.5];
    ax.XColor = [0.8 0.8 0.8];
    ax.YColor = [0.8 0.8 0.8];
    
    % Set labels with appropriate colors
    xlabel('Feature Group', 'Color', [0.9 0.9 0.9]);
    ylabel('Normalized Value', 'Color', [0.9 0.9 0.9]);
    
    % Create legend with dark mode colors
    leg = legend('Query Face', 'Database Face');
    leg.TextColor = [0.9 0.9 0.9];
    leg.Color = [0.25 0.25 0.25];
end

function validName = cleanNameForFilename(name)
    % Replace invalid filename characters with underscores
    invalidChars = ['/', '\', ':', '*', '?', '"', '<', '>', '|', ' '];
    validName = name;
    
    for i = 1:length(invalidChars)
        validName = strrep(validName, invalidChars(i), '_');
    end
end

% Custom histogram equalization function
function outputImage = myHistEq(inputImage)
    % Calculate histogram
    histCounts = zeros(1, 256);
    [rows, cols] = size(inputImage);
    totalPixels = rows * cols;
    
    % Count occurrences of each intensity
    for i = 1:rows
        for j = 1:cols
            intensity = inputImage(i, j) + 1; % +1 because MATLAB is 1-indexed
            histCounts(intensity) = histCounts(intensity) + 1;
        end
    end
    
    % Calculate cumulative distribution
    cdf = zeros(1, 256);
    cdf(1) = histCounts(1);
    for i = 2:256
        cdf(i) = cdf(i-1) + histCounts(i);
    end
    
    % Normalize CDF to range [0,255]
    cdf = uint8((cdf / totalPixels) * 255);
    
    % Apply transformation to each pixel
    outputImage = zeros(rows, cols, 'uint8');
    for i = 1:rows
        for j = 1:cols
            intensity = inputImage(i, j) + 1;
            outputImage(i, j) = cdf(intensity);
        end
    end
end

% Custom Gaussian smoothing function
function smoothedImage = myGaussianSmooth(inputImage, sigma)
    % Define kernel size based on sigma (usually 3*sigma on each side)
    kernelSize = 2 * ceil(3*sigma) + 1;
    
    % Create Gaussian kernel
    kernel = zeros(kernelSize);
    center = floor(kernelSize/2) + 1;
    
    for i = 1:kernelSize
        for j = 1:kernelSize
            x = i - center;
            y = j - center;
            kernel(i, j) = exp(-(x^2 + y^2)/(2*sigma^2));
        end
    end
    
    % Normalize kernel to sum to 1
    kernel = kernel / sum(kernel(:));
    
    % Perform convolution manually for smoothing
    [rows, cols] = size(inputImage);
    paddedImage = padarray(double(inputImage), [floor(kernelSize/2), floor(kernelSize/2)], 'replicate');
    smoothedImage = zeros(rows, cols);
    
    for i = 1:rows
        for j = 1:cols
            % Extract image patch
            patch = paddedImage(i:i+kernelSize-1, j:j+kernelSize-1);
            % Apply kernel
            smoothedImage(i, j) = sum(sum(patch .* kernel));
        end
    end
    
    smoothedImage = uint8(smoothedImage);
end

% Function to detect skin regions
function skinMask = detectSkinRegions(rgbImage)
    % Convert to YCbCr color space
    ycbcrImage = rgb2ycbcr(rgbImage);
    
    % Extract components
    Y = ycbcrImage(:,:,1);
    Cb = ycbcrImage(:,:,2);
    Cr = ycbcrImage(:,:,3);
    
    % Multi-color space approach
    % YCbCr thresholds
    skinMask1 = (Cb >= 77 & Cb <= 127) & (Cr >= 133 & Cr <= 173);
    
    % Also check in RGB space for better accuracy
    R = double(rgbImage(:,:,1));
    G = double(rgbImage(:,:,2));
    B = double(rgbImage(:,:,3));
    
    % RGB rules for skin detection
    skinMask2 = (R > 95) & (G > 40) & (B > 20) & ...
                (max(R, max(G, B)) - min(R, min(G, B)) > 15) & ...
                (abs(R - G) > 15) & (R > G) & (R > B);
    
    % Combine both masks
    skinMask = skinMask1 & skinMask2;
end

% Function to detect eye pairs in potential face regions
function hasEyes = detectEyePair(faceRegion)
    % Resize for consistent processing
    [rows, cols] = size(faceRegion);
    if rows > 100 && cols > 100
        faceRegion = imresize(faceRegion, [100, 100]);
    elseif rows < 20 || cols < 20
        % Too small, unlikely to be a face
        hasEyes = false;
        return;
    end
    
    % Enhance contrast in the face region
    enhancedFace = myHistEq(faceRegion);
    
    % Apply edge detection to find features
    edges = myEdgeDetection(enhancedFace);
    
    % Look in upper half of face for eyes
    upperHalf = edges(1:round(size(edges, 1)/2), :);
    
    % Count edge pixels in the upper region
    edgePixels = sum(upperHalf(:));
    
    % Check if there are enough edge features in the upper face region
    % This is a simplified approach - eyes typically have strong edges
    threshold = 0.03 * numel(upperHalf); % Adjust as needed
    
    hasEyes = edgePixels > threshold;
end

% Simple edge detection function
function edges = myEdgeDetection(image)
    % Simple Sobel edge detection
    sobelX = [-1 0 1; -2 0 2; -1 0 1];
    sobelY = [-1 -2 -1; 0 0 0; 1 2 1];
    
    % Apply Sobel operators
    [rows, cols] = size(image);
    paddedImage = padarray(double(image), [1, 1], 'replicate');
    edgeX = zeros(rows, cols);
    edgeY = zeros(rows, cols);
    
    for i = 1:rows
        for j = 1:cols
            % Extract 3x3 patch
            patch = paddedImage(i:i+2, j:j+2);
            % Apply Sobel
            edgeX(i, j) = sum(sum(patch .* sobelX));
            edgeY(i, j) = sum(sum(patch .* sobelY));
        end
    end
    
    % Compute edge magnitude
    edgeMag = sqrt(edgeX.^2 + edgeY.^2);
    
    % Threshold
    threshold = 30; % Adjust as needed
    edges = edgeMag > threshold;
end

% Non-maximum suppression to handle overlapping detections
function selectedIndices = nonMaximumSuppression(regionProps, validIndices)
    n = length(validIndices);
    
    % If only one region, return it
    if n <= 1
        selectedIndices = validIndices;
        return;
    end
    
    % Get all bounding boxes and areas
    boxes = zeros(n, 4);
    areas = zeros(n, 1);
    
    for i = 1:n
        idx = validIndices(i);
        bbox = regionProps(idx).BoundingBox;
        boxes(i,:) = bbox;
        areas(i) = bbox(3) * bbox(4);
    end
    
    % Sort by area (descending)
    [~, order] = sort(areas, 'descend');
    
    % Initialize mask for selected boxes
    keep = true(1, n);
    
    % Process boxes in order of decreasing area
    for i = 1:n-1
        if keep(order(i))
            % Get current box
            currBox = boxes(order(i),:);
            
            % Check overlap with remaining boxes
            for j = i+1:n
                if keep(order(j))
                    % Get comparison box
                    compBox = boxes(order(j),:);
                    
                    % Calculate overlap
                    x1 = max(currBox(1), compBox(1));
                    y1 = max(currBox(2), compBox(2));
                    x2 = min(currBox(1)+currBox(3), compBox(1)+compBox(3));
                    y2 = min(currBox(2)+currBox(4), compBox(2)+compBox(4));
                    
                    w = max(0, x2-x1);
                    h = max(0, y2-y1);
                    overlap = w * h;
                    
                    % Calculate overlap ratio
                    overlapRatio = overlap / min(currBox(3)*currBox(4), compBox(3)*compBox(4));
                    
                    % If overlap is significant, suppress the smaller box
                    if overlapRatio > 0.5
                        keep(order(j)) = false;
                    end
                end
            end
        end
    end
    
    % Return indices of kept boxes
    selectedIndices = validIndices(order(keep(order)));
end