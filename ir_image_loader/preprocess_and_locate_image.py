# import the necessary packages
import numpy as np
import imutils
import cv2

from PIL import Image


class preprocess_ir_image():
    def __init__(self, image_path, heated_templates):
        self.image_path = image_path
        self.heated_templates  = heated_templates
        
    def remove_bg(self, new_im, im):
        r = new_im.any(1)
        if r.any():
            m,n = new_im.shape
            c = new_im.any(0)
            im = im[r.argmax():m-r[::-1].argmax(), c.argmax():n-c[::-1].argmax()]
        else:
            im = np.empty((0,0),dtype=bool)
        
        return im
        
    def change_median_channel(self, im_copy):
        
        r,g,b = im_copy.split()
        
        b = np.array(b)
        b[b==np.median(b)] = 0
        b = Image.fromarray(b)
        
        return Image.merge('RGB', (r,g,b))

    def crop_2_halves(self, filepath):
    	im = Image.open(filepath) # open the image file
    	im.seek(0) # getting the IR image
    	im_copy = im.copy()
    
    	w,h = im.size
    	im1 = im_copy.crop((0,0,w//2,h))
    	im2 = im_copy.crop((w//2,0,w,h))
    
    	return [im1, im2]

    def get_heated_region(self):
        image_  = self.crop_2_halves(self.image_path)
        found   = None
        
        new_im_ = []
        
        for im, temp in zip(image_, self.heated_templates):
            template = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
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
            new_im_.append(self.change_median_channel(image_heat_region))            
            
        return new_im_

    def process_image(self):
        h1, h2 = self.get_heated_region()
        
        h1_bw = np.array(h1.convert('L'))
        h2_bw = np.array(h2.convert('L'))
        
        h1 = self.remove_bg((h1_bw>0).astype('uint8'), np.array(h1))
        h2 = self.remove_bg((h2_bw>0).astype('uint8'), np.array(h2))
        
        return Image.fromarray(h1), Image.fromarray(h2)