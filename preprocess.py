from PIL import Image, ImageStat
import numpy as np
import glob
import os
import pickle
from sklearn.model_selection import train_test_split


filepath 	= '/media/ankit/ampkit/metric_space/xxx/good/'
res_folder 	= '/media/ankit/ampkit/metric_space/precon_data/io'

def crop_2_halves(filepath):
	im = Image.open(filepath) # open the image file
	im.seek(0) # getting the IR image
	im_copy = im.copy()

	try:
		r,g,b = im_copy.split()
	except:
		r,g,b,a = im_copy.split()

	im_copy = Image.merge('RGB', (r, g, r))

	if 'mask' in res_folder:
		r = np.array(r)
		r[r<255] = 0

		im_copy = Image.fromarray(r)

	w,h = im.size
	im1 = im_copy.crop((0,0,w//2,h))
	im2 = im_copy.crop((w//2,0,w,h))

	stat1 = ImageStat.Stat(im1)
	stat2 = ImageStat.Stat(im2)

	mean_stat = np.array(stat1.mean) + np.array(stat2.mean)
	std_stat = np.array(stat1.stddev) + np.array(stat2.stddev)

	return im1, im2, mean_stat, std_stat


image_files = glob.glob(f'{filepath}/*.tiff')
X_train, X_test = train_test_split(image_files, test_size=0.1, random_state=21)

datasets = [("train", X_train, '/media/ankit/ampkit/metric_space/precon_data/io'),
            ("test", X_test, '/media/ankit/ampkit/metric_space/precon_data/io_test')
            ]

for (dType, keys, outputPath) in datasets:
	for count, image_file in enumerate(keys):
		
		filename = os.path.basename(image_file)
		
		if count == 0:
			h1, h2, mean_stat, std_stat = crop_2_halves(image_file)
		else:
			h1, h2, mean_stat0, std_stat0 = crop_2_halves(image_file)
			mean_stat += mean_stat0
			std_stat += std_stat0

		h1.save(f"{outputPath}/{filename.replace('.tiff', '_1.png')}")
		h2.save(f"{res_folder}/{filename.replace('.tiff', '_2.png')}")
	
	if dType == 'train':
		pickle.dump(mean_stat/len(keys), open('mean_val.pkl', 'wb'))
		pickle.dump(std_stat/len(keys), open('std_dev.pkl', 'wb'))