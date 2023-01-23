from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import time
import cv2

import aiofiles
import uvicorn

import numpy as np
from pathlib import Path

from PIL import Image

from heat_anomaly.deploy import Inferencer
from importlib import import_module
from typing import Optional

from heat_anomaly.config import get_configurable_parameters


app = FastAPI()
IMAGEDIR    = 'precon_web_folder/'
try:
    os.makedirs(IMAGEDIR)
except:
    pass

config = get_configurable_parameters(config_path='./heat_anomaly/models/cflow/ir_image.yaml')
# definintion to load the model
def get_inferencer(config_path: Path, weight_path: Path, meta_data_path: Optional[Path] = None) -> Inferencer:

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    inferencer: Inferencer
    module = import_module("heat_anomaly.deploy")
    torch_inferencer = getattr(module, "TorchInferencer")
    inferencer = torch_inferencer(config=config_path, model_source=weight_path, meta_data_path=meta_data_path)

    return inferencer

def preprocess(filepath):
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

	return np.array(im_copy)

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
    
    inferencer.predict(image=np.array(h1))
    inferencer.predict(image=np.array(h2))

    return True
	
# file_model = 'precon_heatmap.ckpt'
file_model = 'results/cflow/folder/weights/model.ckpt'
inferencer = get_inferencer('./heat_anomaly/models/cflow/ir_image.yaml', file_model)

@app.post("/file")
async def _file_upload( my_file: UploadFile = File(...),
                programmNr: str = Form(...),
        		second: str = Form("default value for second"),
			):
    file_text = os.path.basename(my_file.filename)
    async with aiofiles.open(f"{IMAGEDIR}{file_text}", 'wb') as out_file:
        content = await my_file.read()  # async read
        await out_file.write(content)  # async write
    
    print(f'{file_text}')
    # print(f'program_type:{programmNr}')

    st = time.time()
    h1 = preprocess(f'{IMAGEDIR}{file_text}')
    print('preprocessing done')
    
    predictions = inferencer.predict(image=h1)

    print(predictions.pred_score)

    # checking if anomaly exists
    # -----------------------------
    anomaly = predictions.pred_mask
    is_anomalous = False

    if anomaly.max() > 0:
        is_anomalous = True
    print(f"message: anomalous:{bool(is_anomalous)}")
    
    print('predictions generated')
    res_image   = concat_result_top_down(Image.fromarray(predictions.segmentations), 
                                            Image.fromarray(predictions.heat_map))
    
    print('heated regions merged')
    res_image.save('sample_output.png')
    response_file = 'sample_output.png'
    print(f'elapsed time -> {time.time()-st}')

    return FileResponse(response_file, media_type="image/png", filename=file_text.replace('.tiff','.png'),headers={"message": f"anomalous={bool(is_anomalous)}"})

fill_buffer()

# uvicorn.run(app, host="192.168.5.135", port=8000)
uvicorn.run(app, port=8000)
