from PIL import Image, ImageStat
import numpy as np
import glob
import os
import shutil
from sklearn.model_selection import train_test_split


filepath 	= '/media/ankit/ampkit/metric_space/xxx/good/'
res_folder 	= '/media/ankit/ampkit/metric_space/precon_data/io'

def crop_2_halves(filepath):
	im = Image.open(filepath) # open the image file
	im.seek(0) # getting the IR image
	im_copy = im.copy()
	print(im.size)

	# change to RGB image
	im_copy = im_copy.convert('RGB')
	r,g,b = im_copy.split()
	
	b = np.array(b)
	b[b==np.median(b)] = 0
	b = Image.fromarray(b)
	
	im_copy = Image.merge('RGB', (r,g,b))

	return im_copy


image_files = glob.glob(f'{filepath}/*.tiff')
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
		h1 = crop_2_halves(image_file)

		h1.save(f"{outputPath}/{filename.replace('.tiff', '.png')}")