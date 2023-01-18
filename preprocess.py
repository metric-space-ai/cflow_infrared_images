from PIL import Image
import numpy as np
import glob

def crop_2_halves(filepath):
	im = Image.open(filepath) # open the image file
	im.seek(0) # getting the IR image
	im_copy = im.copy()
	w,h = im.size
	im1 = im_copy.crop((0,0,w//2,h))
	im2 = im_copy.crop((w//2,0,w,h))
	
	dim_hr = ((w//2,h), (w//2,h))
	
	return im1, im2

filepath = '/media/nvidia/ampkit/metric_space/IRBilder/ground_truth/fault'

image_files = glob.glob(f'{filepath}/*.png')

for image_file in image_files:
    h1, h2 = crop_2_halves(image_file)
    h1.save(image_file.replace('.png', '_1.png'))
    h2.save(image_file.replace('.png', '_2.png'))
