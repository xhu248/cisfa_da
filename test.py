import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    data_dir = "../data/mmwhs/ct/longitude"
    output_dir = "tmp"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    files = os.listdir(data_dir)
    for f in files:
        print("saving fig of ", f)
        data_path = os.path.join(data_dir, f)
        img_data = np.load(data_path)[:, 0]
        index = int(img_data.shape[0] * 0.5)

        plt.imshow(img_data[index], cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, "longitude_" + f.split(".")[0] + "_" + str(index) + '.png'))
        plt.close()

