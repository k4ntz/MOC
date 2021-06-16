import cv2
import os
import time

import numpy as np

from os import listdir
from os.path import isfile, join

path = "/data/ATARI/"
# just change to the game you want to create your greyscale images
game = "Pong-v0"
#game = "Tennis-v0"

suffix = "-black"

# folders inside game folder
folders = ["train/" , "test/", "validation/"]

for folder in folders:
    mypath = os.getcwd() + path + game + "/" + folder 
    print("Loading '" + mypath + "' ...")
    start = time.time()
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath,f))]
    for n in range(0, len(onlyfiles)):
        # looping over every image inside this folder
        img_path = join(mypath,onlyfiles[n])
        tmp_img = cv2.imread(img_path)
        # get most dominant color
        colors, count = np.unique(tmp_img.reshape(-1,tmp_img.shape[-1]), axis=0, return_counts=True)
        most_dominant_color = colors[count.argmax()]
        # create the mask and use it to change the colors
        bounds_size = 20
        lower = most_dominant_color - [bounds_size, bounds_size, bounds_size]
        upper = most_dominant_color + [bounds_size, bounds_size, bounds_size]
        mask = cv2.inRange(tmp_img, lower, upper)
        tmp_img[mask != 0] = [0,0,0]
        # dilation 
        kernel = np.ones((3,3), np.uint8)
        img_dilation = cv2.dilate(tmp_img, kernel, iterations=1)
        # show image
        cv2.imshow("Changed image", img_dilation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # save in new folder
        saving_img_path = img_path.replace(game, game + suffix)
        cv2.imwrite(saving_img_path, ttmp_img)
    elapsed_time_fl = (time.time() - start) 
    print("Copying took " + str(elapsed_time_fl) + " sec.")


