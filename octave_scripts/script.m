
clear all

%get folder of good images
datafolder = '/media/ankit/ampkit/metric_space/precon_data/nio_part';
Files = dir(fullfile(datafolder, '*.png'));

%loop through the images
for file_idx = 1:length(Files) 
    disp(['Processing: ' Files(file_idx).name ' (' num2str(file_idx) '/' num2str(length(Files)) ')'])
    filename=strcat(Files(file_idx).folder, '/', Files(file_idx).name);
    I1 = imread(filename);

    imageSizeX = size(I1,2);
    imageSizeY = size(I1,1);

    % get the background color of the blue channel and replace it with
    % Zeros.
    m_blue= median(median(I1(:,:,3))); 
    for i=1:imageSizeY
        for j=1:imageSizeX
            if I1(i,j,3)==m_blue
                I1(i,j,3)=0;
            end
        end
    end
    
    % converting to grayscale and back.
    I1g = rgb2gray(I1);

    % generate dimensions of the anomaly as circle
    minR=randi([2,4]);
    maxR=minR+randi([2,10]);
    
    % search of a spot to place the anomaly
    Summe=0;
    while Summe < 2000
      [columnsInImage, rowsInImage] = meshgrid(1:imageSizeX, 1:imageSizeY);

      centerX = randi(imageSizeX);
      centerY = randi(imageSizeY);
      radius = minR;
      circlePixels = (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 <= radius.^2;
      Summe = sum(I1g(circlePixels));
    end
    
    % generate the granularity of the gradient of the anomaly
    steps=randi([10,20]); 
    radi = round(linspace(minR,maxR,steps));
    
    % generate the anomaly
    B = I1g; % Initialize
    for i=1:length(radi)
        circlePixels = (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 <= radi(i).^2;
        B(circlePixels) = B(circlePixels)*(1-1./length(radi)*round(steps/4));
    end
    B=(recolor(B,I1));
    
    % plots for debugging
    figure(1)
    subplot(2,2,1)
    image(I1)
    title('original')
    subplot(2,2,2)
    image(B)
    title('modified')
    subplot(2,2,4)
    image(circlePixels)
    colormap([0 0 0; 1 1 1]);
    title('mask')
    
    % store the results by increasing index.
    imwrite(B,['abnormal_' num2str(file_idx) '.png'])
    imwrite(circlePixels,[0 0 0; 1 1 1],['abnormal_' num2str(file_idx) '_mask.png'])

end


function rslt=recolor(img,ref)
% This function converts a gray image to RGB based on the colors of a
% source image. Based on the paper "Transfering Color to Grayscale Images" 
% by Welsh, Ashikhmin and Mueller.
%
% imt - the image, that you want to color 
% [normalized floating points (0 ... 1) 2d-array]
%
% ims - reference color image you want to color the grayscale to
% [normalized floating points (0 ... 1) 3d-array]


[tx, ty, tz] = size(img); % get size of target image
[~, ~, sz] = size(ref); % get 3rd dim of source
if tz ~= 1 % convert the destination image to grayscale if not already
    img = rgb2gray(img);
end
if sz ~= 3 % check to see that the source image is RGB
    disp ('img2 must be a color image (not indexed)');
else
    img(:, :, 2) = img(:, :, 1); % add green channel to grayscale img
    img(:, :, 3) = img(:, :, 1); % add blue channel to grayscale img
   
% Converting to ycbcr color space
    % ycbcr, y: luminance, cb: blue difference chroma, cr: red difference chroma
    % s - source, t - target
    nspace1 = rgb2ycbcr(ref); % convert source img to ycbcr color space
    nspace2 = rgb2ycbcr(img); % convert target img to ycbcr color space
    
    % Get unique values of the luminance
    [ms, ics, ~] = unique(double(nspace1(:, :, 1))); % luminance of src img
    mt = unique(double(nspace2(:, :, 1))); % luminance of target img
    % Establish values for the cb and cr content from the source
    % image
    cbs = nspace1(:, :, 2);
    cbs = cbs(ics);
    crs = nspace1(:, :, 3);
    crs = crs(ics);
    
    % get max and min luminance of src and target
    m1 =max(ms);
    m2 = min(ms);
    m3 = max(mt);
    m4 = min(mt);
    d1 = m1 - m2; % get difference between max and min luminance
    d2 = m3 - m4;
    % Normalization 
    dx1 = ms;
    dx2 = mt;
    dx1 = (dx1 * 255) / (255 - d1); % normalize source
    dx2 = (dx2 * 255) / (255 - d2); % normalize target
    [mx, ~] = size(dx2);
    % luminance and normalization of target image
    nimage_norm = double(nspace2(:, :, 1));
    nimage_norm =(nimage_norm * 255) / (255 - d2);
    
    % Luminance Comparison
    nimage = nspace2;
    nimage_cb = nimage(:, :, 2);
    nimage_cr = nimage(:, :, 3);
    
    % reshape cb and cr channels to be column vector
    nimage_cb = reshape(nimage_cb, numel(nimage_cb), 1);
    nimage_cr = reshape(nimage_cr, numel(nimage_cr), 1);
    
    % Loop through dx2 luminance values and find location of 
    % corresponding luminance values in nimage_norm. Assign cb and cr 
    % values to nimage's cb and cr channels for matching values
    
    for i = 1:mx
        iy = dx2(i);
        tmp = abs(dx1 - iy); % calculate absolute difference between 
        % specific normalized target luminance value and normalized 
        % source luminance values
        ck = min(tmp);
        % finds min value of absolute diff. between specific 
        % normalized target luminance value and normalized source
        % luminance values
        r = find(tmp == ck); % finds row and column where tmp = ck
        cb = cbs(r, 1); % establish cb value
        cr = crs(r, 1); % establish cr value
        mtch = find(nimage_norm == iy); % find linear indicies of matching
        % luminance values
        nimage_cb(mtch) = cb(1); % set cb values based on matching lum vals
        nimage_cr(mtch) = cr(1); % set cr values based on matching lum vals
    end
    
    % reshape cb and cr channels to original image dimensions
    nimage_cb = reshape(nimage_cb, tx, ty);
    nimage_cr = reshape(nimage_cr, tx, ty);
    % assign cb and cr channelse of output image
    nimage(:, :, 2) = nimage_cb;
    nimage(:, :, 3) = nimage_cr;

    % converting back to RGB color space
    rslt = ycbcr2rgb(nimage);
end

end