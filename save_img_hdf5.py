import h5py
import os

from PIL import Image

import numpy as np

if __name__ == "__main__":

    train_path = os.path.join("Face Dataset", "Train")

    males_dir = os.path.join(train_path, "Male")
    females_dir = os.path.join(train_path, "Female")

    for each_male in os.listdir(males_dir):

        cur_dir = os.path.join(males_dir, each_male)
        imgs = os.listdir(cur_dir)

        img_array = np.zeros((len(imgs), 105, 105, 1))
        for i, each in enumerate(imgs):
            if ".h5" in each:
                continue
            img = Image.open(os.path.join(cur_dir, each)).convert('L')
            img = img.resize((105, 105), Image.ANTIALIAS)
            img = np.asarray(img).astype(np.float64)
            img = img / img.std() - img.mean()

            img_array[i, :, :, 0] = img

            file = h5py.File(os.path.join(cur_dir, each_male+".h5"), "w")
            dataset = file.create_dataset("images", np.shape(img_array), h5py.h5t.STD_U8BE, img_array)

            file.close()

    for each_female in os.listdir(females_dir):

        cur_dir = os.path.join(females_dir, each_female)
        imgs = os.listdir(cur_dir)

        img_array = np.zeros((len(imgs), 105, 105, 1))
        for i, each in enumerate(imgs):
            if ".h5" in each:
                continue
            img = Image.open(os.path.join(cur_dir, each)).convert('L')
            img = img.resize((105, 105), Image.ANTIALIAS)
            img = np.asarray(img).astype(np.float64)
            img = img / img.std() - img.mean()

            img_array[i, :, :, 0] = img

            file = h5py.File(os.path.join(cur_dir, each_female+".h5"), "w")
            dataset = file.create_dataset("images", np.shape(img_array), h5py.h5t.STD_U8BE, img_array)

            file.close()
