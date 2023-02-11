# The imported libraries
import pickle
import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2
import random

from PIL import Image

import albumentations as albu

# inspecting the image
image1 = pickle.load(open('ir_image_loader/h1_template.pkl', 'rb'))
image1 = Image.fromarray(image1)
image11 = image1.convert('L')
image11 = np.array(image11)
# plt.imshow(image1)

image2 = pickle.load(open('ir_image_loader/h2_template.pkl', 'rb'))
image2 = Image.fromarray(image2)
image2 = image2.convert('L')
image2 = np.array(image2)

# masking the image
mask_image1 = (image11 > image11.mean()).astype('uint8')
# mask_image1 = Image.fromarray(mask_image1*255)

# masking the image
mask_image2 = (image2 > image2.mean()).astype('uint8')
# mask_image2 = Image.fromarray(mask_image2*255)

size = random.randint(4, 8)
starting_point = random.randint(0, len(np.argwhere(mask_image1==1)))

created_range   = np.argwhere(mask_image1==1)[starting_point]
creating_a_rect = ((max(0, created_range[0] - size//2), created_range[1] - size//2), (created_range[0] + size//2, created_range[1] + size//2))

start_pt = (max(0, created_range[0] - size//2), max(0, created_range[1] - size//2))
end_pt = (min(mask_image1.shape[0], created_range[0] + size//2), min(mask_image1.shape[1], created_range[1] + size//2))

center_coordinates = (created_range[1], created_range[0])

image_masks = np.array(image1)

bad_contours_mask   = np.zeros(mask_image1.shape, dtype="uint8")
# bad_contours_mask = cv2.rectangle(bad_contours_mask, start_pt, end_pt, (255, 255, 255), -1)
bad_contours_mask = cv2.circle(bad_contours_mask, center_coordinates, size, (255, 255, 255), -1)

plt.figure(0)
plt.imshow(bad_contours_mask, 'gray')

# loop over the contours
bad_contours = np.argwhere(bad_contours_mask==255)
anomaly_mask   = np.ones(mask_image1.shape, dtype="uint8")
for cr in bad_contours:
    if mask_image1[cr[0], cr[1]] == 1:
        print(mask_image1[cr[0], cr[1]])
    
        anomaly_mask[cr[0], cr[1]] = 0

res_image = anomaly_mask[..., None] * image_masks
plt.figure(1)
plt.imshow(res_image)

mask_res = cv2.bitwise_not(anomaly_mask*255)
plt.figure(2)
plt.imshow(mask_res, 'gray')
