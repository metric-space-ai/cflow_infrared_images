
clear all
pkg load image

%get folder of good images
%datafolder = uigetdir('data','Wähle einen Ordner mit tiff Dateien');
datafolder = '/media/ankit/ampkit/metric_space/precon_data/io_test';
Files = dir(fullfile(datafolder, '*.png'));


%loop through the images
for file_idx = 1:length(Files) 
    filename = Files(file_idx).name;
    disp(['Processing: ' filename ' (' num2str(file_idx) '/' num2str(length(Files)) ')'])
    filename=[datafolder Files(file_idx).name];
    disp(filename)
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
    I2=recolor(I1g,I1);

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
    imwrite(circlePixels.*1,['abnormal_' num2str(file_idx) '_mask.png'])

end



