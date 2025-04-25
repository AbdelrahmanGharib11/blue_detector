function classify_image_simple(imagePath)
    % Read the image
    img = imread(imagePath);
    
    % Process the image with manual skin detection
    result = detectFaceManual(img);
    
    % Save the output using the same pattern as your backend expects
    save_output_image(imagePath, result);
    
    % Display detection result in console
    disp('✅ Manual Face Detection Completed');
end

function result = detectFaceManual(inputImg)
    %@Input -> Takes input an RGB image
    %@returns -> A processed image with extracted features present
    
    % Checks if the image is not a three-channel color image
    if size(inputImg, 3) ~= 3
        result = inputImg;
        return;
    end

    % Convert to YCbCr
    ycbcr = rgb2ycbcr(inputImg);
    cb = ycbcr(:, :, 2);
    cr = ycbcr(:, :, 3);
   
    % Choose a threshold for skin color
    skinMask = (cb >= 77 & cb <= 127) & (cr >= 133 & cr <= 173);

    % Filtering the mask using median filter
    skinMask = medfilt2(skinMask, [5 5]);
    % Filling binary holes for smoothness 
    skinMask = imfill(skinMask, 'holes');
    % Filter blobs that are less than 500 pixels in area
    skinMask = bwareaopen(skinMask, 500);
    
    % Find regions and draw bounding boxes
    stats = regionprops(skinMask, 'BoundingBox');
    result = inputImg;
    
    % Loop through all detected regions
    for i = 1:length(stats)
        box = stats(i).BoundingBox;
        ratio = box(3)/box(4);
        
        % Face detection criteria
        if box(3) > 60 && box(4) > 60 && ratio > 0.6 && ratio < 1.8
            % Draw face bounding box
            result = insertShape(result, 'Rectangle', box, ...
                'Color', 'green', 'LineWidth', 3);
            
            % Add label
            result = insertText(result, [box(1), box(2)-25], 'Face', ...
                'FontSize', 14, 'BoxColor', 'green');
        end
    end
end

function save_output_image(originalPath, img)
    % ROBUST IMAGE SAVING WITH PATH VALIDATION
    
    % 1. Validate input path
    if nargin < 1 || isempty(originalPath) || ~ischar(originalPath) && ~isstring(originalPath)
        % Generate default filename if input is invalid
        originalPath = 'detected_output.png';
        warning('Invalid path provided. Using default filename: %s', originalPath);
    end
    
    % Convert to char if it is a string
    originalPath = char(originalPath);
    
    % 2. Parse path components
    [folder, name, ext] = fileparts(originalPath);
    
    % Handle empty folder case
    if isempty(folder)
        folder = pwd; % Use current directory
    end
    
    % 3. Create results directory
    result_folder = fullfile(folder, 'results');
    if ~exist(result_folder, 'dir')
        try
            mkdir(result_folder);
        catch
            error('Failed to create output directory: %s', result_folder);
        end
    end
    
    % 4. Generate output filename
    if isempty(name)
        name = 'detected_output'; % Default base name
    end
    
    if isempty(ext)
        ext = '.png'; % Default extension
    end
    
    outName = fullfile(result_folder, [name, '_output', ext]);
    
    % 5. Validate the output path
    if isempty(outName)
        error('Generated output path is empty');
    end
    
    % 6. Convert and save image (from previous fixes)
    try
        % Convert image to proper format
        if islogical(img)
            img = uint8(img) * 255;
        elseif isfloat(img)
            if max(img(:)) <= 1
                img = uint8(img * 255);
            else
                img = uint8(img);
            end
        elseif ~isinteger(img)
            img = im2uint8(img);
        end
        
        % Save the image
        imwrite(img, outName);
        fprintf('Successfully saved output to: %s\n', outName);
        
    catch ME
        % Final fallback - try saving to current directory
        try
            fallbackPath = fullfile(pwd, [name, '_output', ext]);
            imwrite(img, fallbackPath);
            warning('Saved to fallback location: %s', fallbackPath);
        catch
            error('Failed to save image: %s\nOriginal error: %s', fallbackPath, ME.message);
        end
    end
end
