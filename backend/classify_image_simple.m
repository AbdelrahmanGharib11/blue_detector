function classify_image_simple(imagePath)    % اقرأ الصورة
    img = imread(imagePath);
    gray = rgb2gray_if_needed(img);

    % اعادة تحجيم
    gray = imresize(gray, [300 300]);
    img = imresize(img, [300 300]);

    % -------- FACE DETECTION --------
    faceDetector = vision.CascadeObjectDetector();
    faceBox = step(faceDetector, img);

    if ~isempty(faceBox)
        disp('✅ Detected: FACE');
        img = draw_boxes(img, faceBox, 'Face');

        % Eye detection
        eyeDetector = vision.CascadeObjectDetector('EyePairBig');
        eyeBox = step(eyeDetector, img);
        if ~isempty(eyeBox)
            eyeColor = estimateColor(img, eyeBox(1,:));
            fprintf('👁️ Eye Color: %s\n', eyeColor);
            img = draw_boxes(img, eyeBox, ['Eyes: ', eyeColor]);
        end

        % Hair detection (فوق الوجه)
        hairBox = faceBox(1,:);
        hairBox(2) = max(1, hairBox(2) - round(hairBox(4) * 0.6));
        hairBox(4) = round(hairBox(4) * 0.5);
        hairColor = estimateColor(img, hairBox);
        fprintf('💇 Hair Color: %s\n', hairColor);
        img = draw_boxes(img, hairBox, ['Hair: ', hairColor]);
        
        % حفظ الصورة بعد الكشف
        save_output_image(imagePath, img);
        return;
    end

    % -------- HAND DETECTION --------
    %handDetector = vision.CascadeObjectDetector('Hand');
    %handBox = step(handDetector, img);
    %if ~isempty(handBox)
        %disp('🖐️ Detected: HAND');
        %img = draw_boxes(img, handBox, 'Hand');
       % save_output_image(imagePath, img);
        %return;
    %end

    % -------- FINGERPRINT DETECTION --------
    edges = edge(gray, 'Canny');
    edgeDensity = sum(edges(:)) / numel(edges);

    if edgeDensity > 0.15
        disp('✅ Detected: FINGERPRINT');
        img = insertText(img, [10 10], 'Fingerprint Detected', 'FontSize', 18, 'BoxColor', 'green');
        save_output_image(imagePath, img);
        return;
    end

    % -------- OTHER OBJECT --------
    disp('ℹ️ Detected: OTHER OBJECT');
    img = insertText(img, [10 10], 'Other Object Detected', 'FontSize', 18, 'BoxColor', 'yellow');
    save_output_image(imagePath, img);


    % ====== HELPERS ======

function gray = rgb2gray_if_needed(img)
    if size(img,3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
end

function imgOut = draw_boxes(img, boxes, label)
    imgOut = img;
    for i = 1:size(boxes,1)
        imgOut = insertShape(imgOut, 'Rectangle', boxes(i,:), 'Color', 'green', 'LineWidth', 3);
        imgOut = insertText(imgOut, boxes(i,1:2), label, 'FontSize', 16, 'BoxColor', 'green');
    end
end

function colorName = estimateColor(img, box)
    region = imcrop(img, box);
    if isempty(region), colorName = 'Unknown'; return; end
    avgColor = mean(reshape(double(region), [], 3), 1);
    r = avgColor(1); g = avgColor(2); b = avgColor(3);
    if r > 150 && g < 80 && b < 80
        colorName = 'Red';
    elseif r < 80 && g > 150 && b < 80
        colorName = 'Green';
    elseif r < 80 && g < 80 && b > 150
        colorName = 'Blue';
    elseif all(avgColor > 180)
        colorName = 'White/Gray';
    elseif all(avgColor < 70)
        colorName = 'Black/Dark';
    elseif r > g && r > b
        colorName = 'Brown/Red';
    elseif g > r && g > b
        colorName = 'Greenish';
    elseif b > r && b > g
        colorName = 'Bluish';
    else
        colorName = 'Mixed';
    end
end

if ~exist('imagePath', 'var') || isempty(imagePath)
    imagePath = 'detected_output.png'; % Provide default
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






end