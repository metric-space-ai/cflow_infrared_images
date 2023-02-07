
pkg load image

%get folder of good images
%datafolder = uigetdir('data','Wähle einen Ordner mit tiff Dateien');
datafolder = '/media/ankit/ampkit/metric_space/precon_data/testing';
Files = dir(fullfile(datafolder, '*.png'));

nio_dir = '/media/ankit/ampkit/metric_space/precon_data/nio'
nio_mask = '/media/ankit/ampkit/metric_space/precon_data/nio_mask'

%loop through the images
for file_idx = 1:length(Files) 
    disp(['Processing: '  ' (' num2str(file_idx) '/' num2str(length(Files)) ')'])
    filename=strcat(datafolder, '/', Files(file_idx).name);
    disp(filename)
    I1 = imread(filename);

    imageSizeX = size(I1,2);
    imageSizeY = size(I1,1);
    
    % converting to grayscale and back.
    I1g = rgb2gray(I1);
    I2=recolor(I1g,I1);

    % generate dimensions of the anomaly as circle
    minR=randi([2,4]);
    maxR=minR+randi([3,5]);
    
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
    steps=randi([20,50]); 
    radi = round(linspace(minR,maxR,steps));
    
    % generate the anomaly
    B = I1g; % Initialize
    for i=1:length(radi)
        circlePixels = (rowsInImage - centerY).^2 + (columnsInImage - centerX).^2 <= radi(i).^2;
        B(circlePixels) = B(circlePixels)*(1-1./length(radi)*round(steps/4));
    end
    B=(recolor(B,I1));
    
    num = 0
    % store the results by increasing index.
    imwrite(B,strcat(nio_dir, '/abnormal_', num2str(num+file_idx), '.png'))
    imwrite(circlePixels.*1,strcat(nio_mask, '/abnormal_', num2str(num+file_idx), '.png'))

end