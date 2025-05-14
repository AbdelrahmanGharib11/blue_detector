function output_path = detect_face_manual(input_path)
    % DETECT2 Process an image for face detection
    % Usage:
    %   output_path = detect_face_manual(input_path)
    %
    % Inputs:
    %   input_path - Full path to the input image file
    %
    % Outputs:
    %   output_path - Full path to the processed output image file
    
    % Check if input file exists
    if ~exist(input_path, 'file')
        error('Input file does not exist: %s', input_path);
    end
    
    try
        % Read the input image
        img = imread(input_path);
        
        % Run the face detection algorithm
        result = detectFaceManual(img);
        
        % Create output filename
        [filepath, name, ext] = fileparts(input_path);
        output_path = fullfile(filepath, [name, '_output', ext]);
        
        % Save the processed image
        imwrite(result, output_path);
        
        % Print the output path for the Flask app to capture
        fprintf('OUTPUT_PATH:%s\n', output_path);
    catch e
        % Handle errors properly
        fprintf('ERROR: Face detection failed - %s\n', e.message);
        rethrow(e);
    end
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
    
    % Find the biggest and most suitable connected components resembling
    % human face or hand
    stats = regionprops(skinMask, 'BoundingBox');
    result = inputImg;
    
    % Loop the stats struct for each connected components and draw
    % suitable rectangles around each
    for i = 1:length(stats)
        box = stats(i).BoundingBox;
        ratio = box(3)/box(4);
        if box(3) > 60 && box(4) > 60 && ratio > 0.6 && ratio < 1.8
            result = insertShape(result, 'Rectangle', box, ...
                'Color', 'green', 'LineWidth', 3);
        end
    end
end