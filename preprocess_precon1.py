# import the necessary packages
from matplotlib import pyplot as plt
import numpy as np
import imutils
import glob
import cv2

def get_heated_region():
    

def process_image(filepath):
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

	return im_copy

def plt_imshow(title, image):
	# convert the image frame BGR to RGB color space and display it
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	plt.imshow(image)
	plt.title(title)
	plt.grid(False)
	plt.show()

args = {
	"template": ["/media/ankit/ampkit/metric_space/precon_data/example_mask/left_heated_region.png",
              '/media/ankit/ampkit/metric_space/precon_data/example_mask/right_heated_region.png'],
	"images": "/media/ankit/ampkit/metric_space/precon_data/test_locating"
}

coordinates_heated_region = []
for imagePath in glob.glob(args["images"] + "/*.tiff"):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    
    for temp in args['template']:
        template = cv2.imread(temp)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        (tH, tW) = template.shape[:2]
        # plt_imshow("Template", template)
        
        # loop over the scales of the image
        for scale in np.linspace(0.2, 1.0, 20)[::-1]:
            resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
            r = gray.shape[1] / float(resized.shape[1])
            
            
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break
            
            edged = cv2.Canny(resized, 50, 200)
            result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
            
            if args.get("visualize", False):
                clone = np.dstack([edged, edged, edged])
                cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                              (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
                cv2.imshow("Visualize", clone)
                cv2.waitKey(0)
            
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
        
        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        
        coordinates_heated_region.append((startX, startY, endX, endY))
        
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        plt_imshow("Image", image)

