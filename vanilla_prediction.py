
import time
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
    parser.add_argument("--image", type=str, 
                        default='/media/ankit/ampkit/metric_space/xxx/good/b38f5c32863420c53fe6d3819e4ad63849f694d605a6a40d00afbff02c6107e5_3a9d32fb90b0ad429bff3f0a1b8501ca5b8dc7210c0c024c855ae520c96d143b.tiff', 
                        help="Path to image(s) to infer.")

    args = parser.parse_args()

    return args

args = get_args()

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

def fill_buffer():
    temp_image = Image.open('cache_input.tiff')

    w,h = temp_image.size
    h1 = temp_image.crop((0,0,w//2,h))
    h2 = temp_image.crop((w//2,0,w,h))
    print('preprocessing done')
    
    inferencer_h1.predict(image=np.array(h1))
    inferencer_h2.predict(image=np.array(h2))

    return True

file_model_h1 = 'results_left/cflow/folder/weights/model.ckpt'
file_model_h2 = 'results_right/cflow/folder/weights/model.ckpt'
inferencer_h1 = get_inferencer('./heat_anomaly/models/cflow/ir_image_left.yaml', file_model_h1)
inferencer_h2 = get_inferencer('./heat_anomaly/models/cflow/ir_image_right.yaml', file_model_h2)	

# creating the model once
st = time.time()

#preprocess_image
pir = preprocess_ir_image(args.image)
h1, h2 = pir.process_image()
print('preprocessing done')

predictions_h1 = inferencer_h1.predict(image=np.array(h1))
predictions_h2 = inferencer_h2.predict(image=np.array(h2))

print(predictions_h1.pred_score, predictions_h2.pred_score)

# checking if anomaly exists
# -----------------------------
anomaly_h1 = predictions_h1.pred_mask
anomaly_h2 = predictions_h2.pred_mask
is_anomalous = False

if anomaly_h1.max() > 0 or anomaly_h2.max():
    is_anomalous = True
print(f"message: anomalous:{bool(is_anomalous)}")

print('predictions generated')
res_image1  = concat_result(Image.fromarray(predictions_h1.segmentations).resize(h1.size), Image.fromarray(predictions_h2.segmentations).resize(h1.size))
res_image2  = concat_result(Image.fromarray(predictions_h1.heat_map).resize(h1.size), Image.fromarray(predictions_h2.heat_map).resize(h1.size))
res_image   = concat_result_top_down(res_image1, res_image2)

print(time.time() - st)
