import os
import fnmatch
import random

import numpy as np

from batchgenerators.dataloading import SlimDataLoaderBase
from datasets.data_loader import MultiThreadedDataLoader
from .data_augmentation import get_transforms

# get three parameters file (directory of processed images), files_len, slcies_ax( list of tuples)
def load_dataset(base_dir, pattern='*.npy', slice_offset=5, keys=None):
    fls = []
    files_len = []
    slices_ax = []

    for root, dirs, files in os.walk(base_dir):
        i = 0
        for filename in sorted(fnmatch.filter(files, pattern)):

            if keys is not None and filename in keys:
                npy_file = os.path.join(root, filename)
                numpy_array = np.load(npy_file, mmap_mode="r+")  # change "r" to "r+"

                fls.append(npy_file)
                files_len.append(numpy_array.shape[0]) # changed from 0 to 1

                slices_ax.extend([(i, j) for j in range(slice_offset, files_len[-1] - slice_offset)])

                i += 1

    return fls, files_len, slices_ax,


class NumpyDataSet(object):
    """
    TODO
    """
    def __init__(self, x_base_dir, y_base_dir, mode="train", batch_size=16, num_batches=10000000, seed=None,
                 num_processes=8, num_cached_per_queue=8 * 4, target_size=128,
                 file_pattern='*.npy', label_slice=1, input_slice=(0,), do_reshuffle=True, x_keys=None, y_keys=None):

        data_loader = NumpyXYDataLoader(x_base_dir=x_base_dir, y_base_dir=y_base_dir, mode=mode, batch_size=batch_size,
                                        num_batches=num_batches, seed=seed, file_pattern=file_pattern,
                                        input_slice=input_slice, label_slice=label_slice, x_keys=x_keys, y_keys=y_keys)

        self.data_loader = data_loader
        self.batch_size = batch_size
        self.do_reshuffle = do_reshuffle
        self.number_of_slices = 1

        self.transforms = get_transforms(mode=mode, target_size=target_size)
        self.augmenter = MultiThreadedDataLoader(data_loader, self.transforms, num_processes=num_processes,
                                                 num_cached_per_queue=num_cached_per_queue, seeds=seed,
                                                 shuffle=do_reshuffle)
        self.augmenter.restart()

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        if self.do_reshuffle:
            self.data_loader.reshuffle()
        self.augmenter.renew()
        return self.augmenter

    def __next__(self):
        return next(self.augmenter)


class NumpyXYDataLoader(SlimDataLoaderBase):
    def __init__(self, x_base_dir, y_base_dir, mode="train", batch_size=16, num_batches=10000000,
                 seed=None, file_pattern='*.npy', label_slice=1, input_slice=(0,), x_keys=None, y_keys=None):

        self.x_files, self.x_file_len, self.x_slices = load_dataset(base_dir=x_base_dir, pattern=file_pattern,
                                                                    slice_offset=0, keys=x_keys, )

        self.y_files, self.y_file_len, self.y_slices = load_dataset(base_dir=y_base_dir, pattern=file_pattern,
                                                                    slice_offset=0, keys=y_keys, )
        super(NumpyXYDataLoader, self).__init__(self.x_slices, batch_size, num_batches)

        self.transforms = get_transforms(mode=mode)

        self.batch_size = batch_size

        self.use_next = False
        if mode == "train":
            self.use_next = False

        self.data_len = min(len(self.x_slices), len(self.y_slices))

        self.slice_idxs = list(range(0, self.data_len))  # divide 3D images into slices

        self.num_batches = min((self.data_len // self.batch_size)+10, num_batches)

        if isinstance(label_slice, int):
            label_slice = (label_slice,)
        self.input_slice = input_slice
        self.label_slice = label_slice

        self.np_x_data = np.asarray(self.x_slices)
        self.np_y_data = np.array(self.y_slices)

    def reshuffle(self):
        print("Reshuffle...")
        random.shuffle(self.slice_idxs)
        print("Initializing... this might take a while...")

    def generate_train_batch(self):
        open_arr = random.sample(self._data, self.batch_size)
        return self.get_data_from_array(open_arr, open_arr)

    def __len__(self):
        n_items = min(self.data_len // self.batch_size, self.num_batches)
        return n_items

    def __getitem__(self, item):
        slice_idxs = self.slice_idxs
        data_len = self.data_len

        if item > len(self):
            raise StopIteration()
        if (item * self.batch_size) == data_len:
            raise StopIteration()

        start_idx = (item * self.batch_size) % data_len
        stop_idx = ((item + 1) * self.batch_size) % data_len

        if ((item + 1) * self.batch_size) == data_len:
            stop_idx = data_len

        if stop_idx > start_idx:
            idxs = slice_idxs[start_idx:stop_idx]
        else:
            raise StopIteration()

        open_arr_x = self.np_x_data[idxs]  # tuple (a,b) of images of this batch
        open_arr_y = self.np_y_data[idxs]

        return self.get_data_from_array(open_arr_x, open_arr_y)

    def get_data_from_array(self, open_arr_x, open_arr_y):
        x_data = []
        x_fnames = []
        x_slice_idxs = []
        x_labels = []
        ret_dict = {}

        for slice in open_arr_x:
            # slice is a tuple (a,b), slice[0] indicating which image it's,
            # and slice[1] incicats which one in the 3d image it's.
            x_fn_name = self.x_files[slice[0]]

            x_array = np.load(x_fn_name, mmap_mode="r")  # load data from .npy to numpy_arrary

            x_slice = x_array[slice[1]]  #(2,64,64)

            x_data.append(x_slice[None, self.input_slice[0]])   # 'None' keeps the dimension   (1,64,64)

            if self.label_slice is not None:
                x_labels.append(x_slice[None, self.label_slice[0]])   # 'None' keeps the dimension

            x_fnames.append(self.x_files[slice[0]])
            x_slice_idxs.append(slice[1])

        x_labels = np.asarray(x_labels)
        x_labels = np.asarray(x_labels)
        ret_dict_A = {'data': np.asarray(x_data), 'fnames': x_fnames,
                      'slice_idxs': x_slice_idxs}

        if self.label_slice is not None:
            ret_dict_A['seg'] = x_labels

        ret_dict_A = self.transforms(**ret_dict_A)
        ret_dict["A"] = ret_dict_A["data"]
        ret_dict["A_paths"] = ret_dict_A["fnames"]
        ret_dict["B_slice_idxs"] = ret_dict_A["slice_idxs"]
        ret_dict["segA"] = ret_dict_A["seg"]

        y_data = []
        y_fnames = []
        y_slice_idxs = []
        y_labels = []

        for slice in open_arr_y:
            # slice is a tuple (a,b), slice[0] indicating which image it's,
            # and slice[1] incicats which one in the 3d image it's.
            y_fn_name = self.y_files[slice[0]]

            y_array = np.load(y_fn_name, mmap_mode="r")  # load data from .npy to numpy_arrary

            y_slice = y_array[slice[1]]  # (2,64,64)

            y_data.append(y_slice[None, self.input_slice[0]])  # 'None' keeps the dimension   (1,64,64)

            if self.label_slice is not None:
                y_labels.append(y_slice[None, self.label_slice[0]])  # 'None' keeps the dimension

            y_fnames.append(self.y_files[slice[0]])
            y_slice_idxs.append(slice[1])

        y_labels = np.asarray(y_labels)
        ret_dict.update({'B': np.asarray(y_data), 'B_paths': y_fnames,
                    'B_slice_idxs': y_slice_idxs})  # data_shape (8,1,64,64) 'data': np.asarray(data),

        if self.label_slice is not None:
            ret_dict['segB'] = y_labels

        return ret_dict



