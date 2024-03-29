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

import numpy as np
from batchgenerators.transforms import Compose, MirrorTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform, RandomCropTransform
from batchgenerators.transforms.spatial_transforms import ResizeTransform, SpatialTransform, Rot90Transform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.color_transforms import BrightnessTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

from torchvision import transforms


def get_transforms(mode="train", target_size=192):
    tranform_list = []

    if mode == "train":
        tranform_list = [ # Rot90Transform(p_per_sample=0.5, axes=(0, 1)),
                         # NumpyToTensor(),
        ]

    elif mode == "val":
        tranform_list = [# CenterCropTransform(crop_size=target_size),

                         ]

    elif mode == "test":
        tranform_list = [# CenterCropTransform(crop_size=target_size),

                         ]

    elif mode == "simclr":
        tranform_list = [
            BrightnessTransform(mu=1, sigma=1, p_per_sample=0.5),
            GammaTransform(p_per_sample=0.5),
            GaussianNoiseTransform(p_per_sample=0.5),
            SpatialTransform(patch_size=(target_size, target_size), random_crop=True,
                             do_elastic_deform=True, alpha=(0., 1000.), sigma=(40., 60.),
                             do_rotation=True, p_rot_per_sample=0.5,
                             angle_z=(0, 2 * np.pi),
                             scale=(0.7, 1.25), p_scale_per_sample=0.5,
                             border_mode_data="nearest", border_mode_seg="nearest"),
            NumpyToTensor(),
        ]

        return TwoCropTransform(Compose(tranform_list))

    tranform_list.append(NumpyToTensor())

    return Compose(tranform_list)


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, **x):
        return [self. transform(**x), self.transform(**x)]
   