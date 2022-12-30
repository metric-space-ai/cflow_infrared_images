from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import time

app = FastAPI()

IMAGEDIR = '/home/ankit/Documents/precon_web_folder/'
import aiofiles

import numpy as np
from pathlib import Path

from PIL import Image

from heat_anomaly.deploy import Inferencer
from importlib import import_module
from typing import Optional

from heat_anomaly.config import get_configurable_parameters

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

def crop_2_halves(filepath):
	im = Image.open(filepath) # open the image file
	im.seek(0) # getting the IR image
	im_copy = im.copy()
	w,h = im.size
	im1 = im_copy.crop((0,0,w//2,h))
	im2 = im_copy.crop((w//2,0,w,h))
	
	dim_hr = ((w//2,h), (w//2,h))
	
	return np.array(im1), np.array(im2), dim_hr

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

inferencer = get_inferencer('./heat_anomaly/models/cflow/ir_image.yaml', 'results/cflow/folder/weights/model-v1_copy.ckpt')

# Code from: https://fastapi.tiangolo.com/tutorial/request-files/
app = FastAPI()

@app.post("/file")
async def _file_upload( my_file: UploadFile = File(...),
        		first: str = Form(...),
        		second: str = Form("default value for second"),
			):
    file_text = os.path.basename(my_file.filename)
    async with aiofiles.open(f"{IMAGEDIR}{file_text}", 'wb') as out_file:
        content = await my_file.read()  # async read
        await out_file.write(content)  # async write
    
    st = time.time()
    h1,h2, dims = crop_2_halves(f'{IMAGEDIR}{file_text}')
    print('preprocessing done')
    
    predictions1 = inferencer.predict(image=h1)
    predictions2 = inferencer.predict(image=h2)
    
    print('predictions generated')
    res_image1  = concat_result(Image.fromarray(predictions1.segmentations).resize(dims[0]), Image.fromarray(predictions2.segmentations).resize(dims[0]))
    res_image2  = concat_result(Image.fromarray(predictions1.heat_map).resize(dims[0]), Image.fromarray(predictions2.heat_map).resize(dims[0]))
    res_image   = concat_result_top_down(res_image1, res_image2)
    
    print('heated regions merged')
    res_image.save('sample_output.png')
    response_file = 'sample_output.png'
    print(f'elapsed time -> {time.time()-st}')

    return FileResponse(response_file, media_type="image/png", filename=file_text.replace('.tiff','.png'))

import uvicorn
# uvicorn.run(app, host="192.168.8.113", port=8000)
uvicorn.run(app, port=8000)