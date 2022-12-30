
from PIL import Image
import numpy as np
import cv2
import os
from argparse import ArgumentParser, Namespace
from heat_anomaly.config import get_configurable_parameters
from heat_anomaly.models import get_model
from heat_anomaly.utils.callbacks import get_callbacks
from pytorch_lightning import Trainer

import time
from torch.utils.data import DataLoader
from heat_anomaly.data.inference import InferenceDataset

import cv2
import io
import numpy as np
from pathlib import Path

import torch
import cv2
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from heat_anomaly.deploy import Inferencer
from importlib import import_module
from typing import Optional, Tuple

from heat_anomaly.config import get_configurable_parameters

def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--image", type=str, default='/media/ankit/ampkit/metric_space/xxx/good/b38f5c32863420c53fe6d3819e4ad63849f694d605a6a40d00afbff02c6107e5_3a9d32fb90b0ad429bff3f0a1b8501ca5b8dc7210c0c024c855ae520c96d143b.tiff', 
                        help="Path to image(s) to infer.")

    args = parser.parse_args()

    return args

args = get_args()

def get_heated_region(filepath):
    '''
    
    Parameters
    ----------
    filepath : image file with .tif extension 
        DESCRIPTION.

    Returns
    -------
    boxed_region : the corners to get the accurate region
        DESCRIPTION.
        
    '''
    # opening image and grayscale conversion
    # --------------------
    im = Image.open(filepath) # open the image file
    im.seek(0) # getting the IR image
    imL = im.convert('L') # grayscale conversion
    
    # mask the image
    # ----------------
    image = np.array(imL)
    mask = (image > image.mean()).astype(np.uint8)*255
    
    #dilate the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    
    heated_region = []
    dim_hr = []
    for count_cnt, cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        dim_hr.append((w,h))

        cropped_region = im.crop((x,y, x+w, y+h))
        cropped_region = cropped_region.resize((256, 256), Image.Resampling.LANCZOS)
        heated_region.append(cropped_region)
    
    return np.array(heated_region[0]), np.array(heated_region[1]), dim_hr


config = get_configurable_parameters(config_path='./anomalib/models/cflow/ir_image.yaml')
# definintion to load the model
def get_inferencer(config_path: Path, weight_path: Path, meta_data_path: Optional[Path] = None) -> Inferencer:
    """Parse args and open inferencer.

    Args:
        config_path (Path): Path to model configuration file or the name of the model.
        weight_path (Path): Path to model weights.
        meta_data_path (Optional[Path], optional): Metadata is required for OpenVINO models. Defaults to None.

    Raises:
        ValueError: If unsupported model weight is passed.

    Returns:
        Inferencer: Torch or OpenVINO inferencer.
    """

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    inferencer: Inferencer
    module = import_module("anomalib.deploy")
    torch_inferencer = getattr(module, "TorchInferencer")
    inferencer = torch_inferencer(config=config_path, model_source=weight_path, meta_data_path=meta_data_path)

    return inferencer

def concat_result(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

# creating the model once
st = time.time()
inferencer = get_inferencer('./anomalib/models/cflow/ir_image.yaml', 'results/cflow/folder/weights/model-v2.ckpt')

#preprocess the image
img1,img2, dims = get_heated_region(args.image)

predictions1 = inferencer.predict(image=img1)
predictions2 = inferencer.predict(image=img2)

res_image = concat_result(Image.fromarray(predictions1.segmentations).resize(dims[0]),
                            Image.fromarray(predictions2.segmentations).resize(dims[0]))

res_image.save('sample_output.png')

print(time.time() - st)
