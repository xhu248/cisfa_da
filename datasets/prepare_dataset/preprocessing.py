#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from batchgenerators.augmentations.utils import resize_image_by_padding

from medpy.io import load
import os
import numpy as np
import shutil
import torch
import torch.nn.functional as F


def preprocess_data(root_dir):
    image_dir = os.path.join(root_dir, 'images')
    label_dir = os.path.join(root_dir, 'labels')
    output_dir = os.path.join(root_dir, 'orig')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    class_stats = defaultdict(int)
    total = 0
    nii_files = subfiles(image_dir, suffix=".nii.gz", join=False)

    total_shape = []
    for f in nii_files:
        if f.startswith("."):
            os.remove(os.path.join(image_dir, f))
            continue
        file_dir = os.path.join(output_dir, f.split('.')[0]+'.npy')
        if not os.path.exists(file_dir) or os.path.exists(file_dir):
            print(f)
            image, _ = load(os.path.join(image_dir, f))
            label, _ = load(os.path.join(label_dir, f.replace('image', 'label')))


            # normalize images
            image = (image - image.min()) / (image.max() - image.min())
            print(image.shape, label.shape)

            # image = image[:,:,0].transpose((0, 2, 1))
            # label = label.transpose(0, 2, 1)

            total_shape.append(image.shape)


            # modify the label
            label[label == 500] = 1
            label[label == 600] = 2
            label[label == 420] = 3
            label[label == 421] = 3  # a special case for mri
            label[label == 550] = 4
            label[label == 205] = 5
            label[label == 820] = 6
            label[label == 850] = 7
            print(label.max())

            result = np.stack((image, label)).transpose((3, 0, 1, 2))  # ct image
            # result = np.stack((image, label)).transpose((2, 0, 1, 3))  # mri image
            print(result.shape)
            np.save(os.path.join(output_dir, f.split('.')[0] + '.npy'), result)

    # total_shape = np.array(total_shape)
    # print(total_shape[:, 0].max(), total_shape[:, 0].min(), total_shape[:, 0].mean())
    # print(total_shape[:, 1].max(), total_shape[:, 1].min(), total_shape[:, 1].mean())
    # print(total_shape[:, 2].max(), total_shape[:, 2].min(), total_shape[:, 2].mean())


def padding_imgs(orig_img, append_value=-1024, new_shape=(512, 512, 512)):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = 0  # (new_shape[1] - orig_img.shape[1]) // 2
    z_offset = 0  # (new_shape[2] - orig_img.shape[2]) // 2

    reshaped_image[x_offset:orig_img.shape[0]+x_offset, y_offset:orig_img.shape[1]+y_offset, z_offset:orig_img.shape[2]+z_offset] = orig_img
    # insert temp_img.min() as background value

    return reshaped_image


def reshape_2d_data(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    files_list = os.listdir(input_dir)

    for f in files_list:
        target_factor = os.path.join(output_dir, f)
        data = np.load(os.path.join(input_dir, f))

        image = data[:, 0]
        label = data[:, 1]

        target_size[0] = image.shape[0]

        if image.shape != target_size:
            image_tensor = torch.from_numpy(image)
            label_tensor = torch.from_numpy(label)

            new_image = F.interpolate(image_tensor[None, None], size=target_size, mode="trilinear")
            new_image = new_image.squeeze().cpu().numpy()

            new_label = F.interpolate(label_tensor[None, None], size=target_size, mode="trilinear")
            new_label = new_label.squeeze().cpu().numpy()

            new_data = np.concatenate((new_image[None], new_label[None]))
            new_data = new_data.transpose(1, 0, 2, 3)
            # new_data = new_data.transpose(2, 0, 1, 3)  # mri image
            print(new_data.shape)
            np.save(target_factor, new_data)


def reshape_3d_data(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    files_list = os.listdir(input_dir)

    for f in files_list:
        target_path = os.path.join(output_dir, f)
        if not os.path.exists(target_path) or os.path.exists(target_path):
            data = np.load(os.path.join(input_dir, f))

            image = data[:, 0]
            label = data[:, 1]

            if image.shape != target_size:
                image_tensor = torch.from_numpy(image)
                label_tensor = torch.from_numpy(label)

                new_image = F.interpolate(image_tensor[None, None], size=target_size, mode="trilinear")
                new_image = new_image.squeeze().cpu().numpy()

                new_label = F.interpolate(label_tensor[None, None], size=target_size, mode="trilinear")
                new_label = new_label.squeeze().cpu().numpy()

                new_data = np.concatenate((new_image[None], new_label[None]))
                print(new_data.shape)
                np.save(target_path, new_data)
            else:
                new_data = data.transpose(1, 0, 2, 3)
                print(new_data.shape)
                np.save(target_path, new_data)


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y   # lambda is another simplified way of defining a function
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
            and (prefix is None or i.startswith(prefix))
            and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def generate_false_labels(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

        files_list = os.listdir(input_dir)

        for f in files_list:
            target_path = os.path.join(output_dir, f)
            data = np.load(os.path.join(input_dir, f))

            image = data[:, 0]
            label = data[:, 1]

            label_copy = label.copy()
            label[label_copy == 1] = 2
            label[label_copy == 2] = 3
            label[label_copy == 3] = 5
            label[label_copy == 5] = 1

            new_data = np.concatenate((image[None], label[None]))
            new_data = new_data.transpose(1, 0, 2, 3)

            print(new_data.shape)
            np.save(target_path, new_data)


def get_crop_range(label):
    # crop the data to only contain the aoi
    shape = label.shape
    # for dim-0
    bottom1 = 0
    upper1 = 0
    for k in range(shape[0]):
        label_sum = np.sum(label[k])
        if bottom1 == 0 and label_sum != 0:
            bottom1 = k
        if k + 1 > upper1 and label_sum != 0:
            upper1 = k + 1

    bottom2 = 0
    upper2 = 0
    for k in range(shape[1]):
        label_sum = np.sum(label[:, k])
        if bottom2 == 0 and label_sum != 0:
            bottom2 = k
        if k + 1 > upper2 and label_sum != 0:
            upper2 = k + 1

    bottom3 = 0
    upper3 = 0
    for k in range(shape[2]):
        label_sum = np.sum(label[:, :, k])
        if bottom3 == 0 and label_sum != 0:
            bottom3 = k
        if k + 1 > upper3 and label_sum != 0:
            upper3 = k + 1
    return bottom1, upper1, bottom2, upper2, bottom3, upper3


def crop_image(img, label, x1, x2, y1, y2, z1, z2):
    new_image = img[x1:x2, y1:y2, z1:z2]
    new_label = label[x1:x2, y1:y2, z1:z2]

    return new_image, new_label


def crop_pipeline(input_dir, output_dir, target_size):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('Created' + output_dir + '...')

    files_list = os.listdir(input_dir)

    for f in files_list:
        target_path = os.path.join(output_dir, f)
        data = np.load(os.path.join(input_dir, f))

        image = data[:, 0]
        label = data[:, 1]
        slice_number = image.shape[0]

        print(f, image.shape)

        if image.max() > 1:
            print(f)
            image = image / 1024

        # image = (2*image - image.min() - image.max()) / (image.max() - image.min())

        x1, x2, y1, y2, z1, z2 = get_crop_range(label)
        x1, y1, z1 = int(x1), int(0.9*y1), int(0.9*z1)  # add some margin to the border, except for the slice axis
        x2, y2, z2 = min(slice_number, int(x2)), min(image.shape[1], int(1.1*y2)), min(image.shape[2], int(1.1*z2))
        print("crop image size:", x2-x1, y2-y1, z2-z1)
        new_image, new_label = crop_image(image, label, x1, x2, y1, y2, z1, z2)
        new_image = (2*new_image - new_image.min() - new_image.max()) / (new_image.max() - new_image.min())
        print(new_image.max(), new_image.min(), np.average(new_image))

        target_size[0] = new_image.shape[0]

        if new_image.shape != target_size:
            image_tensor = torch.from_numpy(new_image)
            label_tensor = torch.from_numpy(new_label)

            new_image = F.interpolate(image_tensor[None, None], size=target_size, mode="trilinear")
            new_image = new_image.squeeze().cpu().numpy()

            new_label = F.interpolate(label_tensor[None, None], size=target_size, mode="trilinear")
            new_label = new_label.squeeze().cpu().numpy()

            new_data = np.concatenate((new_image[None], new_label[None]))
            new_data = new_data.transpose(1, 0, 2, 3)
            # new_data = new_data.transpose(2, 0, 1, 3)  # mri image
            print(new_data.shape)
            np.save(target_path, new_data)


if __name__ == "__main__":
    root_dir = "../../../3d_segmentation/data/mmwhs/mr"
    input_dir = "../../../3d_segmentation/data/mmwhs/mri/orig"
    target_dir = "../../data/mmwhs/mri/cropped"
    target_dir_1 = "../../data/mmwhs/mri/fslabels"
    src_dir_1 = os.path.join(root_dir, "ct_train2")
    k = 5
    j = 10 -k

    # preprocess_data(root_dir)

    # reshape_2d_data(input_dir, target_dir_1, target_size=[128, 128, 128])

    crop_pipeline(input_dir, target_dir, [192, 192, 192])

