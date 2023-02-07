# import the necessary packages
import numpy as np
import imutils
import glob
import cv2

from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
import os


class preprocess_ir_image():
    def __init__(self):
        self.image_path = image_path

# change to RGB image
def change_median_channel(im_copy):
    r,g,b = im_copy.split()
    
    b = np.array(b)
    b[b==np.median(b)] = 0
    b = Image.fromarray(b)
    
    return Image.merge('RGB', (r,g,b))

def crop_2_halves(filepath):
	im = Image.open(filepath) # open the image file
	im.seek(0) # getting the IR image
	im_copy = im.copy()

	w,h = im.size
	im1 = im_copy.crop((0,0,w//2,h))
	im2 = im_copy.crop((w//2,0,w,h))

	return [im1, im2]

def get_heated_region(imagePath, heated_templates):
    image_ = crop_2_halves(imagePath)
    found = None
    
    new_im_ = []
    
    for im, temp in zip(image_, heated_templates):
        template = cv2.imread(temp)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]
        
        gray    = np.array(im.convert('L'))
        
        for scale in np.linspace(1, 1.0, 5)[::-1]:
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            
            
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc  , r)
        
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + (tH+50)) * r))
        
        image_heat_region = im.crop((startX, startY, endX, endY))
        new_im_.append(change_median_channel(image_heat_region))
        
        
        
    return new_im_
    
    

def process_image(filepath, heated_templates):
    left_side, right_side = get_heated_region(filepath, heated_templates)
    
    return left_side, right_side


image_files = glob.glob('/media/ankit/ampkit/metric_space/precon_data/new_data/*.tiff')
X_train, X_test = train_test_split(image_files, test_size=0.1, random_state=21)
X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=21)

datasets = [("train", X_train, '/media/ankit/ampkit/metric_space/precon_data/io'),
            ("val", X_test, '/media/ankit/ampkit/metric_space/precon_data/nio_part'),
			("test", X_test, '/media/ankit/ampkit/metric_space/precon_data/io_test')
            ]

for (dType, keys, outputPath) in datasets:
    try:
        shutil.rmtree(outputPath)
    except:
        pass
        
    os.makedirs(outputPath)
    for count, image_file in enumerate(keys):
        filename = os.path.basename(image_file)
        h1, h2 = process_image(image_file, ["/media/ankit/ampkit/metric_space/precon_data/example_mask/left_heated_region.png",
                                              '/media/ankit/ampkit/metric_space/precon_data/example_mask/right_heated_region.png'])
        
        h1.save(f"{outputPath}/{filename.replace('.tiff', '_left.png')}")
        h2.save(f"{outputPath}/{filename.replace('.tiff', '_right.png')}")
