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



def crop_2_halves(filepath):
    im = Image.open(filepath) # open the image file
    im.seek(0) # getting the IR image
    im_copy = im.copy()
    
    r,g,b = im_copy.split()
    im_copy = Image.merge('RGB', (r, g, r))
    
    w,h = im_copy.size
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
file_model = 'results/cflow/folder/weights/precon_heatmap.ckpt'
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
    h1,h2, dims = crop_2_halves(f'{IMAGEDIR}{file_text}')

    # h1,h2, dims = get_heated_region(f'{IMAGEDIR}{file_text}')
    print('preprocessing done')
    
    predictions1 = inferencer.predict(image=h1)
    predictions2 = inferencer.predict(image=h2)

    print(predictions1.pred_score)
    print(predictions2.pred_score)

    # checking if anomaly exists
    # -----------------------------
    anomaly1, anomaly2 = predictions1.pred_mask, predictions2.pred_mask
    is_anomalous_left, is_anomalous_right = False, False

    if anomaly1.max() > 0:
        is_anomalous_left = True
    if anomaly2.max() > 0:
        is_anomalous_right = True
    print(f"message: anomalous:{bool(is_anomalous_left + is_anomalous_right)}")
    
    print('predictions generated')
    res_image1  = concat_result(Image.fromarray(predictions1.segmentations).resize(dims[0]), Image.fromarray(predictions2.segmentations).resize(dims[0]))
    res_image2  = concat_result(Image.fromarray(predictions1.heat_map).resize(dims[0]), Image.fromarray(predictions2.heat_map).resize(dims[0]))
    res_image   = concat_result_top_down(res_image1, res_image2)
    
    print('heated regions merged')
    res_image.save('sample_output.png')
    response_file = 'sample_output.png'
    print(f'elapsed time -> {time.time()-st}')

    return FileResponse(response_file, media_type="image/png", filename=file_text.replace('.tiff','.png'),headers={"message": f"anomalous={bool(is_anomalous_left+is_anomalous_right)}"})

fill_buffer()

# uvicorn.run(app, host="192.168.5.135", port=8000)
uvicorn.run(app, port=8000)
