import h5py
import os

from PIL import Image, ImageOps

import numpy as np

if __name__ == "__main__":
    for x in ["Train", "Test"]:
        path = os.path.join("Face Dataset", x)

        males_dir = os.path.join(path, "Male")
        females_dir = os.path.join(path, "Female")

        for each_male in os.listdir(males_dir):

            cur_dir = os.path.join(males_dir, each_male)
            imgs = os.listdir(cur_dir)

            img_array = np.zeros((len(imgs)-1, 105, 105, 3))
            i = 0
            for each in imgs:
                if ".h5" in each:
                    continue
                img = Image.open(os.path.join(cur_dir, each)).convert('RGB')
                img = img.resize((105, 105))
                img = np.array(img).astype('float64')
                img /= 255.0

                img_array[i, :, :, :] = img
                i += 1
            file = h5py.File(os.path.join(cur_dir, each_male+".h5"), "w")
            dataset = file.create_dataset("images", np.shape(img_array), data=img_array, dtype='float64')

            file.close()

        for each_female in os.listdir(females_dir):

            cur_dir = os.path.join(females_dir, each_female)
            imgs = os.listdir(cur_dir)

            img_array = np.zeros((len(imgs)-1, 105, 105, 3))
            i = 0
            for each in imgs:
                if ".h5" in each:
                    continue
                img = Image.open(os.path.join(cur_dir, each)).convert('RGB')
                img = img.resize((105, 105))
                img = np.array(img).astype('float64')
                img /= 255.0
                # img = img / img.std() - img.mean()

                img_array[i, :, :, :] = img
                i += 1

            file = h5py.File(os.path.join(cur_dir, each_female+".h5"), "w")
            dataset = file.create_dataset("images", np.shape(img_array), data=img_array, dtype='float64')

            file.close()
