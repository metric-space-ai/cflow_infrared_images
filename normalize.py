# -*- coding: utf-8 -*-

file = '/media/ankit/ampkit/metric_space/precon_data/Datenpool/2023-01-18/untrained.tiff'

from PIL import Image
import albumentations as A
import numpy as np
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2

im = Image.open(file)
im_copy = im.copy()
im_copy = im_copy.convert('RGB')

im_copy = im_copy.convert('RGB')
r,g,b   = im_copy.split()


b               = np.array(b)
b[b==np.median(b)] = 0
b               = Image.fromarray(b)

file_yaml = '/media/ankit/ampkit/metric_space/precon/precon_code/precon/heat_anomaly/models/cflow/transform.yaml'
transforms = A.Compose(
            [
                A.Resize(height=288, width=381, always_apply=True),
                A.CLAHE (clip_limit=4.0, tile_grid_size=(2, 2), always_apply=False, p=1),
                A.HorizontalFlip(p=1),
                # A.ShiftScaleRotate(p=1, border_mode=0, 
                #                    shift_limit_x=[-0.3, 0.3], shift_limit_y=[-0.3, 0.3]),
                A.VerticalFlip(),
                A.Normalize(mean=(0, 0, 0), std=(1,  1,  1), p=1),
                ToTensorV2()
            ]
        )
# A.save(transforms, file_yaml, data_format='yaml')
# image = transforms(image=image)['image']
plt.imshow(im_copy)



