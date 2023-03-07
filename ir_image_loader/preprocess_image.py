# import the necessary packages
# --------------------------------------
import numpy as np
from PIL import Image

class preprocess_ir_image():
    def __init__(self, image_path):
        self.image_path = image_path
    
    # remove the background
    # --------------------------
    def remove_bg(self, new_im, im):
        r = new_im.any(1)
        if r.any():
            m,n = new_im.shape
            c = new_im.any(0)
            im = im[r.argmax():m-r[::-1].argmax(), c.argmax():n-c[::-1].argmax()]
        else:
            im = np.empty((0,0),dtype=bool)
        return im
    
    # make the blue channel to zero
    # ---------------------------------
    def change_blue_channel(self, im_copy):
        r,g,b = im_copy.split()
        
        b = np.array(b)
        b = np.zeros(b.shape, np.uint8)
        b = Image.fromarray(b)
        
        return Image.merge('RGB', (r,g,b))
    
    # substitute the median values to zero
    # ---------------------------------------
    def change_median_channel(self, im_copy):
        r,g,b = im_copy.split()
        
        b = np.array(b)
        b[b==np.median(b)] = 0
        b = Image.fromarray(b)
        
        return Image.merge('RGB', (r,g,b))

    # crop the image in 2 halves
     #--------------------------------
    def crop_2_halves(self, filepath):
        im = Image.open(filepath) # open the image file
        im.seek(0) # getting the IR image
        im_copy = im.copy()
        
        w,h = im.size
        im1 = im_copy.crop((0,0,w//2,h))
        im2 = im_copy.crop((w//2,0,w,h))
        
        return [im1, im2]

    # get the 2 heated region
    #------------------------------
    def get_heated_region(self):
        image_  = self.crop_2_halves(self.image_path)
        new_im_ = []
        
        for im in image_:
            new_im_.append(self.change_median_channel(im))                        
        return new_im_

    # process the image
    # ------------------------
    def process_image(self):
        h1, h2 = self.get_heated_region()
        
        h1_bw = np.array(h1.convert('L'))
        h2_bw = np.array(h2.convert('L'))
        
        h1 = self.remove_bg((h1_bw>0).astype('uint8'), np.array(h1))
        h2 = self.remove_bg((h2_bw>0).astype('uint8'), np.array(h2))
        
        return Image.fromarray(h1), Image.fromarray(h2)
    