import pickle
import os
from .NumpyXYDataLoader import NumpyDataSet


def create_dataset(opt):
    with open(os.path.join(opt.src_dir, "splits.pkl"), 'rb') as f:
        x_splits = pickle.load(f)
    with open(os.path.join(opt.target_dir, "splits.pkl"), 'rb') as f:
        y_splits = pickle.load(f)

    x_tr_keys = x_splits[opt.fold]['train'] + x_splits[opt.fold]['val']
    x_test_keys = x_splits[opt.fold]['test']

    y_tr_keys = y_splits[opt.fold]['train'] + y_splits[opt.fold]['val']
    y_test_keys = y_splits[opt.fold]['test']

    dir_A = os.path.join(opt.src_data_dir,)
    dir_B = os.path.join(opt.target_data_dir)
    train_data_loader = NumpyDataSet(x_base_dir=dir_A, y_base_dir=dir_B, batch_size=opt.batch_size,
                                     x_keys=x_tr_keys, y_keys=y_tr_keys)
    test_data_loader = NumpyDataSet(x_base_dir=dir_A, y_base_dir=dir_B, batch_size=opt.batch_size,
                                     x_keys=x_test_keys, y_keys=y_test_keys)

    return train_data_loader, test_data_loader

