# import the necessary packages
import os
import sys
import glob
import shutil
import pickle
import argparse

from sklearn.model_selection import train_test_split

from ir_image_loader.preprocess_and_locate_image import preprocess_ir_image

from oct2py import octave
octave.addpath(octave.genpath('octave_scripts'))


ap = argparse.ArgumentParser()

ap.add_argument("-b", "--base_path", type=str, 
                default= '/media/ankit/ampkit/metric_space/precon_data',
                help="base folder path")

ap.add_argument("-ri", "--raw_image_path", type=str, default= 'new_data',
                help="path to newly acquired images")

ap.add_argument("-io", "--io_path", type=str, default= 'io',
                help="path to training images")

ap.add_argument("-nio", "--nio_path", type=str, default= 'nio',
                help="path to testing images")

ap.add_argument("-dp", "--delete_previous", type=bool, default= False,
                help="path to testing images")

args = vars(ap.parse_args())


image_files = glob.glob(f"{args['base_path']}/{args['raw_image_path']}/*.tiff")
h1_template = pickle.load(open('ir_image_loader/h1_template.pkl', 'rb'))
h2_template = pickle.load(open('ir_image_loader/h2_template.pkl', 'rb'))

X_train, X_test = train_test_split(image_files, test_size=0.1, random_state=21)
X_val, X_test = train_test_split(X_test, test_size=0.5, random_state=21)

datasets = [("train", X_train, f"{args['base_path']}/{args['io_path']}"),
            ("val", X_test, f"{args['base_path']}/{args['io_path']}_test"),
			("test", X_test, f"{args['base_path']}/{args['nio_path']}_pre_anomalous")
            ]

# making the train, validation and testing split
for (dType, keys, outputPath) in datasets:
    if args['delete_previous']:
        try:
            shutil.rmtree(outputPath)
        except:
            pass
    
    try:
        os.makedirs(outputPath)
    except:
        pass
        
    for count, image_file in enumerate(keys):
        
        sys.stdout.write(f'\r[INFO]: {dType} - {count+1:04d}/{len(keys)}')
    
        filename = os.path.basename(image_file)
        pir = preprocess_ir_image(image_file, [h1_template, h2_template])
        
        h1, h2 = pir.process_image()
        
        h1.save(f"{outputPath}/{filename.replace('.tiff', '_left.png')}")
        h2.save(f"{outputPath}/{filename.replace('.tiff', '_right.png')}")


# creating the anomalies and storing the data
# ----------------------------------------------
nio_folders = [f"{args['base_path']}/{args['nio_path']}", f"{args['base_path']}/{args['nio_path']}_mask"]

if args['delete_previous']:
    for nio_f in nio_folders:
        try:
            shutil.rmtree(nio_f)
        except:
            pass

for nio_f in nio_folders:
    try:
        os.makedirs(nio_f)
    except:
        pass

octave.run('generate_anomalies.m')