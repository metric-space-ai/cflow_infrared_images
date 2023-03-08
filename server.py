from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import time

import aiofiles
import uvicorn

import numpy as np

from PIL import Image
from pathlib import Path
from typing import Optional
from importlib import import_module

from heat_anomaly.deploy import Inferencer
from ir_image_loader.preprocess_image import preprocess_ir_image


app = FastAPI()
IMAGEDIR    = 'precon_web_folder/'
try:
    os.makedirs(IMAGEDIR)
except:
    pass

# definintion to load the model
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
    temp_image = 'ir_image_loader/cache_input.tiff'
    
    pir = preprocess_ir_image(temp_image)
    h1, h2 = pir.process_image()
    
    inferencer_h1.predict(image=np.array(h1))
    inferencer_h2.predict(image=np.array(h2))

    print('buffer filled')

    return True

def get_box_plot_data(bp):
    
    upper_quartile = np.percentile(bp, 75)
    lower_quartile = np.percentile(bp, 25)
    median = np.percentile(bp, 50)

    print(upper_quartile, lower_quartile, median)
    
    if median < 0.3:

        return 0
    
    return upper_quartile - lower_quartile
	
# file_model = 'precon_heatmap.ckpt'
file_model_h1 = 'weights/model_h1.ckpt'
file_model_h2 = 'weights/model_h2.ckpt'
inferencer_h1 = get_inferencer('yaml/ir_image_h1.yaml', file_model_h1)
inferencer_h2 = get_inferencer('yaml/ir_image_h2.yaml', file_model_h2)

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
    print(f'program_type:{programmNr}')
    
    st = time.time()
    try:

        #preprocess_image
        pir = preprocess_ir_image(f'{IMAGEDIR}{file_text}')
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

        anomaly_map_1 = predictions_h1.anomaly_map
        anomaly_map_2 = predictions_h2.anomaly_map

        if anomaly_h1.max() > 0 or anomaly_h2.max()>0:
            is_anomalous = True

        if not is_anomalous:
            bp1 = get_box_plot_data(anomaly_map_1.flatten())
            bp2 = get_box_plot_data(anomaly_map_2.flatten())
            if get_box_plot_data(bp1) > 0.1 or get_box_plot_data(bp2) > 0.1:
                is_anomalous = True
            
        
        print(f"message: anomalous = {bool(is_anomalous)}")
        
        print('predictions generated')
        res_image1  = concat_result(Image.fromarray(predictions_h1.segmentations).resize(h1.size), Image.fromarray(predictions_h2.segmentations).resize(h1.size))
        res_image2  = concat_result(Image.fromarray(predictions_h1.heat_map).resize(h1.size), Image.fromarray(predictions_h2.heat_map).resize(h1.size))
        res_image   = concat_result_top_down(res_image1, res_image2)
        
        print('heated regions merged')
        res_image.save('sample_output.png')
        response_file = 'sample_output.png'
        print(f'elapsed time -> {time.time()-st}')
    
    except:
        im = Image.open(f'{IMAGEDIR}{file_text}')
        im.seek(0)
        im.save('sample_output.png')

        print(f'elapsed time -> {time.time()-st}')
        return FileResponse('sample_output.png', media_type="image/png", filename=file_text.replace('.tiff','.png'),headers={"message": f"anomalous=BAD_FILE"})

    if not is_anomalous:
        os.remove(f'{IMAGEDIR}{file_text}')

    return FileResponse(response_file, media_type="image/png", filename=file_text.replace('.tiff','.png'),headers={"message": f"anomalous={bool(is_anomalous)}"})

fill_buffer()

# uvicorn.run(app, host="192.168.5.135", port=8000)
uvicorn.run(app, port=8000)
