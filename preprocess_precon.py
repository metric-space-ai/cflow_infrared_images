# import the necessary packages
# -----------------------------------
import os
import cv2
import sys
import glob
import shutil
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from ir_image_loader.preprocess_image import preprocess_ir_image

# creating the user input arguments
# ----------------------------------------
ap = argparse.ArgumentParser()

ap.add_argument("-b", "--base_path", type=str, 
                default= '/media/ankit/ampkit/metric_space/precon_data',
                help="base folder path")

ap.add_argument("-s", "--is_subfolder", type=bool, 
                default= False,
                help="Does the raw folder path have subfolders ?")

ap.add_argument("-ri", "--raw_image_path", type=str, default= 'new_data',
                help="path to newly acquired images")

ap.add_argument("-io", "--io_path", type=str, default= 'io',
                help="path to training images")

ap.add_argument("-nio", "--nio_path", type=str, default= 'nio',
                help="path to testing images")

ap.add_argument("-dp", "--delete_previous", type=bool, default= True,
                help="path to testing images")

args = vars(ap.parse_args())

# getting all the tiff files from the path
# ---------------------------------------------
if args["is_subfolder"]:
    image_files = glob.glob(f"{args['base_path']}/{args['raw_image_path']}/**/*.tiff")
else:
    image_files = glob.glob(f"{args['base_path']}/{args['raw_image_path']}/*.tiff")

# forming 2 paths for two halves
# -------------------------------------
h1_path = f"{args['base_path']}/h1"
h2_path = f"{args['base_path']}/h2"

# fragmenting the raw data into 3 parts - training, validation and testing
# -------------------------------------------------------------------------------
X_train, X_test = train_test_split(image_files, test_size=0.1, random_state=21)
X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=21)

# creating the dataset path to loop over 
# ------------------------------------------
datasets = [("train", X_train, f"{args['io_path']}"),
            ("val", X_val, f"{args['io_path']}_test"),
			("test", X_test, f"{args['nio_path']}_pre_anomalous")
            ]

# creating the anomalies and storing the data
# ----------------------------------------------
nio_folders = [args['nio_path'], f"{args['nio_path']}_mask"]

if args['delete_previous']:
    for heated_regio_path in [h1_path, h2_path]:
        for nio_f in nio_folders:
            try:
                shutil.rmtree(f"{heated_regio_path}/{nio_f}")
            except:
                pass

for heated_regio_path in [h1_path, h2_path]:
    for nio_f in nio_folders:
        try:
            os.makedirs(f"{heated_regio_path}/{nio_f}")
        except:
            pass

# making the train, validation and testing split
# ---------------------------------------------------
for (dType, keys, outputPath) in datasets:

    # delete the previously create training folders 
    # ---------------------------------------------------
    if args['delete_previous']:
        for heated_regio_path in [h1_path, h2_path]:
            try:
                shutil.rmtree(f"{heated_regio_path}/{outputPath}")
            except:
                pass
    
    # create the folders if required
    # -----------------------------------
    for heated_regio_path in [h1_path, h2_path]:
        try:
            os.makedirs(f"{heated_regio_path}/{outputPath}")
        except:
            pass
    
    # preprocess the images and save them accordingly
    # --------------------------------------------------
    for count, image_file in enumerate(keys):
        sys.stdout.write(f'\r[INFO]: {dType} - {count+1:04d}/{len(keys)}')
    
        filename = os.path.basename(image_file)
        pir = preprocess_ir_image(image_file)
        
        h1, h2 = pir.process_image()
        
        h1.save(f"{h1_path}/{outputPath}/{filename.replace('.tiff', '.png')}")
        h2.save(f"{h2_path}/{outputPath}/{filename.replace('.tiff', '.png')}")
        
        # create anomalous images for training purposes
        # ----------------------------------------------------
        if dType == 'test':
            for image_anomaly, test_image in zip([h1_path, h2_path], [h1, h2]):
                image11 = test_image.convert('L')
                image11 = np.array(image11)
                
                size        = random.randint(4, 8)
                mask_image1 = (image11 > image11.mean()).astype('uint8')
                starting_point = random.randint(0, len(np.argwhere(mask_image1==1)))
                created_range   = np.argwhere(mask_image1==1)[starting_point]
                center_coordinates = (created_range[1], created_range[0])
                
                image_masks = np.array(test_image)
                
                bad_contours_mask   = np.zeros(mask_image1.shape, dtype="uint8")
                bad_contours_mask = cv2.circle(bad_contours_mask, center_coordinates, size, (255, 255, 255), -1)

                # loop over the contours
                bad_contours = np.argwhere(bad_contours_mask==255)
                anomaly_mask   = np.ones(mask_image1.shape, dtype="uint8")
                for cr in bad_contours:
                    if mask_image1[cr[0], cr[1]] == 1:
                        anomaly_mask[cr[0], cr[1]] = 0

                res_image   = anomaly_mask[..., None] * image_masks
                mask_res    = cv2.bitwise_not(anomaly_mask*255)

                res_image   = Image.fromarray(res_image)
                mask_res    = Image.fromarray(mask_res)
                
                for path_anomaly, generated_anomaly in zip(nio_folders, [res_image, mask_res]):
                    try:
                        os.makedirs(f'{image_anomaly}/{path_anomaly}')
                    except:
                        pass
                    
                    generated_anomaly.save(f"{image_anomaly}/{path_anomaly}/{filename.replace('.tiff', '.png')}")
        
    print('\n')
