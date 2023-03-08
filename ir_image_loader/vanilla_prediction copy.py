import os
import glob
import time
import pickle
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Optional
from importlib import import_module
from heat_anomaly.deploy import Inferencer
from argparse import ArgumentParser, Namespace
from ir_image_loader.preprocess_image import preprocess_ir_image

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, 
                        default='/media/ankit/ampkit/metric_space/precon_data/new_data', 
                        help="Path to image(s) to infer.")
    
    parser.add_argument("--res_path", type=str, 
                        default='/media/ankit/ampkit/metric_space/precon_data/anomaly_map', 
                        help="anomaly map path.")

    args = parser.parse_args()

    return args

args = get_args()

for path_res in [args.res_path, f'{args.res_path}/h1', f'{args.res_path}/h2']:
    try:
        os.makedirs(path_res)
    except:
        pass

def get_inferencer(config_path: Path, weight_path: Path, meta_data_path: Optional[Path] = None) -> Inferencer:
    inferencer: Inferencer
    module = import_module("heat_anomaly.deploy")
    torch_inferencer = getattr(module, "TorchInferencer")
    inferencer = torch_inferencer(config=config_path, model_source=weight_path, meta_data_path=meta_data_path)

    return inferencer

def concat_result(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def concat_result_top_down(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height+im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

file_model_h1 = 'weights/model_h1.ckpt'
file_model_h2 = 'weights/model_h2.ckpt'
inferencer_h1 = get_inferencer('yaml/ir_image_h1.yaml', file_model_h1)
inferencer_h2 = get_inferencer('yaml/ir_image_h2.yaml', file_model_h2)	

# creating the model once
st = time.time()

def get_box_plot_data(bp):
    
    upper_quartile = np.percentile(bp, 75)
    lower_quartile = np.percentile(bp, 25)
    median = np.percentile(bp, 50)

    return upper_quartile - lower_quartile, median

all_image = sorted(glob.glob(f'{args.base_path}/*.tiff'))

value_dict = {}
for count_ai, ai in enumerate(all_image):
    print(f'{count_ai+1:03d}:{os.path.basename(ai)}')
    value_dict[os.path.basename(ai)] = {}
    try:
        #preprocess_image
        pir     = preprocess_ir_image(ai)
        h1, h2  = pir.process_image()
        # print('preprocessing done')

        predictions_h1 = inferencer_h1.predict(image=np.array(h1))
        predictions_h2 = inferencer_h2.predict(image=np.array(h2))

        # print(predictions_h1.pred_score, predictions_h2.pred_score)

        # checking if anomaly exists
        # -----------------------------
        anomaly_h1 = predictions_h1.pred_mask
        anomaly_h2 = predictions_h2.pred_mask
        is_anomalous = False

        anomaly_map_1 = predictions_h1.anomaly_map
        anomaly_map_2 = predictions_h2.anomaly_map

        quart_diff_h1, median_h1 = get_box_plot_data(anomaly_map_1.flatten())
        quart_diff_h2, median_h2 = get_box_plot_data(anomaly_map_2.flatten())

        value_dict[os.path.basename(ai)] = {'median_h1': median_h1, 'interquartile_diff_h1': quart_diff_h1,
                                            'median_h2': median_h2, 'interquartile_diff_h2': quart_diff_h2}

        if anomaly_h1.max() > 0 or anomaly_h2.max():
            is_anomalous = True
        # print(f"message: anomalous:{bool(is_anomalous)}")

        # print('predictions generated')
        res_image1  = concat_result(Image.fromarray(predictions_h1.segmentations).resize(h1.size), Image.fromarray(predictions_h2.segmentations).resize(h1.size))
        res_image2  = concat_result(Image.fromarray(predictions_h1.heat_map).resize(h1.size), Image.fromarray(predictions_h2.heat_map).resize(h1.size))
        res_image   = concat_result_top_down(res_image1, res_image2)

        heatmap_1 = predictions_h1.anomaly_map
        heatmap_2 = predictions_h2.anomaly_map

        np.save(f'{args.res_path}/h1/{os.path.basename(ai).replace(".tiff", ".npy")}', heatmap_1)
        np.save(f'{args.res_path}/h2/{os.path.basename(ai).replace(".tiff", ".npy")}', heatmap_2)
    except Exception as e:
        print(e)
        pass

print(time.time() - st)
pickle.dump(open('dict_threshold.ms', 'wb'), value_dict)