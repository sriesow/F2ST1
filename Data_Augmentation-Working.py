
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage import img_as_ubyte
import cv2
import os
from shutil import copyfile
import glob


# In[ ]:


def getRotatedImage(img, angle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    temp_obj = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, temp_obj, (h, w))


# In[ ]:


def saveImage(dest_path, filename, extension, img):
    file_base_name  = os.path.basename(filename)
    dest_filename = "{}_{}{}".format(os.path.splitext(file_base_name)[0], extension, os.path.splitext(file_base_name)[1])
    cv2.imwrite(os.path.join(dest_path, dest_filename), img)


# In[ ]:


def augment_store_image(filename, dest_images_path):
    saveImage(dest_images_path, filename, "Org", cv2.imread(filename))
    
    img = cv2.imread(filename)
    
    saveImage(dest_images_path, filename, "1", cv2.flip(img, 0))
    saveImage(dest_images_path, filename, "2", cv2.flip(img, 1))
    saveImage(dest_images_path, filename, "3", getRotatedImage(img, 90))
    saveImage(dest_images_path, filename, "4", cv2.flip(getRotatedImage(img, 90), 0))
    saveImage(dest_images_path, filename, "5", cv2.flip(getRotatedImage(img, 90), 1))
    saveImage(dest_images_path, filename, "6", getRotatedImage(img, 180))
    saveImage(dest_images_path, filename, "7", cv2.flip(getRotatedImage(img, 180), 0))
    saveImage(dest_images_path, filename, "8", cv2.flip(getRotatedImage(img, 180), 1))
    saveImage(dest_images_path, filename, "9", getRotatedImage(img, 270))
    saveImage(dest_images_path, filename, "10", cv2.flip(getRotatedImage(img, 270), 0))
    saveImage(dest_images_path, filename, "11", cv2.flip(getRotatedImage(img, 270), 1))

    img_hist_eq = np.concatenate([np.expand_dims(cv2.equalizeHist(img[:,:,i]), axis=2) for i in range(3)], axis=2)
    saveImage(dest_images_path, filename, "12", img_hist_eq)
    saveImage(dest_images_path, filename, "13", cv2.flip(img_hist_eq, 0))
    saveImage(dest_images_path, filename, "14", cv2.flip(img_hist_eq, 1))


# In[ ]:


source_images_path = "/home/ubuntu/Data/AI4AXI/Bad_parts"
dest_images_path = "/home/ubuntu/Data/AI4AXI/Bad_parts_Augmented"


# In[ ]:


i = 0
for file_name in glob.glob(source_images_path + "/*.png"):
    print("Working on {}/200..".format(i+1))
    augment_store_image(file_name, dest_images_path)

