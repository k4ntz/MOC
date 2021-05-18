import cv2
import os
import time

from os import listdir
from os.path import isfile, join

path = "/data/ATARI/"
# just change to the game you want to create your greyscale images
game = "Pong-v0"
#game = "Tennis-v0"

file_suffix = ".jpg"
suffix = "-grey"

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
        # make grey
        gray_img = cv2.cvtColor(tmp_img, cv2.COLOR_BGR2GRAY)
        # show image
        #cv2.imshow("Gray image", gray_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # save in new folder
        saving_img_path = img_path.replace(game, game + suffix)
        cv2.imwrite(saving_img_path, gray_img)
    elapsed_time_fl = (time.time() - start) 
    print("Copying took " + str(elapsed_time_fl) + " sec.")


