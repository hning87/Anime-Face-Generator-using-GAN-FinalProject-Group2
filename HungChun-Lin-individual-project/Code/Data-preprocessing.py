import os
import cv2
import numpy as np

# %% ----------------------------------- Define functions --------------------------------------------------------------
def load_data(image_dir):
    # Load the images.
    RESIZE_TO = 64
    img = []
    for path in [f for f in os.listdir(image_dir) if f[-4:] == ".jpg"]:
        img.append(cv2.cvtColor(cv2.resize(cv2.imread(image_dir + path),
                                           (RESIZE_TO, RESIZE_TO)),
                                cv2.COLOR_BGR2RGB))
    img = np.array(img)

    return img

def load_extra(image_dir):
    # Load the images.
    RESIZE_TO = 64
    img = []
    for path in [f for f in os.listdir(image_dir) if f[-4:] == ".jpg"]:
        img.append(cv2.cvtColor(cv2.resize(cv2.imread(image_dir + path),
                                           (RESIZE_TO, RESIZE_TO)),
                                cv2.COLOR_BGR2RGB))
    img = np.array(img)

    return img

def combine_data(image1, image2):
    imgs = np.concatenate((image1, image2), axis=0)
    return imgs

# %% ----------------------------------- Load data --------------------------------------------------------------
image1 = load_data(os.getcwd() + "/faces/")
image2 = load_extra(os.getcwd() + "/extra_data/images/")
imgs = combine_data(image1, image2)

print(imgs.shape)

np.save("imgs.npy", imgs)

import matplotlib.pyplot as plt
plt.imshow(imgs[0])
plt.show()
