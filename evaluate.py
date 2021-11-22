from datasets.two_dim_multi_stream import create_dataset
from models import create_model
from options.train_options import TrainOptions
from loss_functions.metrics import SegmentationMetric
from loss_functions.dice_loss import SoftDiceLoss

import torch
import torch.nn.functional as F
import os
import numpy as np
import pickle

import medpy.metric.binary as mmb


def evaluate(ckpt_dir):
    opt = TrainOptions().parse()
    split_dir = os.path.join(opt.target_dir, "splits.pkl")
    with open(split_dir, 'rb') as f:
        splits = pickle.load(f)

    test_keys = splits[opt.fold]['test']

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # create model and oad state_dict
    model = create_model(opt)
    seg_model = model.netS_B.to(device)
    netF = model.netF.to(device)
    netG = model.netG.to(device)
    nce_layers = [0, 4, 8, 12, 16]

    state_dict = torch.load(os.path.join(ckpt_dir, "latest_net_S_B.pth"))
    seg_model.load_state_dict(state_dict)
    state_dict = torch.load(os.path.join(ckpt_dir, "latest_net_G.pth"))
    netG.load_state_dict(state_dict)
    criterionSeg = SoftDiceLoss(batch_dice=True, do_bg=False)

    dice_list = []
    assd_list = []
    metric_val = SegmentationMetric(opt.num_classes)

    with torch.no_grad():
        for k in test_keys:
            data_path = os.path.join(opt.target_dir, 'cropped', k)
            volume = np.load(data_path)[:, 0]
            label = np.load(data_path)[:, 1]

            slice_num = volume.shape[0]
            volume_pred = np.zeros(volume.shape)
            for i in range(0, slice_num):
                img = volume[i]
                img = torch.from_numpy(img[None, None]).float().to(device)
                output = seg_model(img)
                pred_softmax = F.softmax(output, dim=1)
                pred = torch.argmax(pred_softmax, dim=1).cpu().numpy()

                volume_pred[i] = pred.squeeze(0)

                metric_val.update(torch.from_numpy(label[i]).unsqueeze(0), pred_softmax)

            for c in range(1, opt.num_classes):
                pred_test_data_tr = volume_pred.copy()
                pred_test_data_tr[pred_test_data_tr != c] = 0
                pred_test_data_tr[pred_test_data_tr == c] = 1

                pred_gt_data_tr = label.copy()
                pred_gt_data_tr[pred_gt_data_tr != c] = 0
                pred_gt_data_tr[pred_gt_data_tr == c] = 1

                dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                if pred_test_data_tr.max() == 1:
                    print(pred_test_data_tr.max(), pred_test_data_tr.min())
                    assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))

    _, _, Dice = metric_val.get()
    print("Overall mean dice score is:", Dice)
    print("Finished test")

    dice_arr = 100 * np.reshape(dice_list, [opt.num_classes - 1, -1])

    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)

    print(dice_mean.shape, dice_arr.shape)
    print('Dice:')
    print('AA :%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.2f' % np.mean(dice_mean))

    assd_arr = np.reshape(assd_list, [opt.num_classes - 1, -1])

    assd_mean = np.mean(assd_arr, axis=1)
    assd_std = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('AA :%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.2f' % np.mean(assd_mean))
    print("Finished test")
    print(ckpt_dir)


def evaluate_test(ckpt_dir):
    opt = TrainOptions().parse()

    data_dir = "../data/sifa_data/ct/new_test"
    # test_keys = splits[opt.fold]['test']
    test_keys = os.listdir((data_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # create model and oad state_dict
    model = create_model(opt)
    seg_model = model.netS_B.to(device)
    netF = model.netF.to(device)
    netG = model.netG.to(device)
    nce_layers = [0, 4, 8, 12, 16]

    state_dict = torch.load(os.path.join(ckpt_dir, "latest_net_S_B.pth"))
    seg_model.load_state_dict(state_dict)
    state_dict = torch.load(os.path.join(ckpt_dir, "latest_net_G.pth"))
    netG.load_state_dict(state_dict)
    criterionSeg = SoftDiceLoss(batch_dice=True, do_bg=False)

    dice_list = []
    assd_list = []
    metric_val = SegmentationMetric(opt.num_classes)

    with torch.no_grad():
        for k in test_keys:
            data_path = os.path.join(data_dir, k)
            volume = np.load(data_path)[:, 0]
            label = np.load(data_path)[:, 1]

            slice_num = volume.shape[0]
            volume_pred = np.zeros(volume.shape)
            for i in range(0, slice_num):
                img = volume[i]
                img = torch.from_numpy(img[None, None]).float().to(device)
                output = seg_model(img)
                pred_softmax = F.softmax(output, dim=1)
                pred = torch.argmax(pred_softmax, dim=1).cpu().numpy()

                volume_pred[i] = pred.squeeze(0)

                metric_val.update(torch.from_numpy(label[i]).unsqueeze(0), pred_softmax)

            for c in range(1, opt.num_classes):
                pred_test_data_tr = volume_pred.copy()
                pred_test_data_tr[pred_test_data_tr != c] = 0
                pred_test_data_tr[pred_test_data_tr == c] = 1

                pred_gt_data_tr = label.copy()
                pred_gt_data_tr[pred_gt_data_tr != c] = 0
                pred_gt_data_tr[pred_gt_data_tr == c] = 1

                dice_list.append(mmb.dc(pred_test_data_tr, pred_gt_data_tr))
                if pred_test_data_tr.max() == 1:
                    assd_list.append(mmb.assd(pred_test_data_tr, pred_gt_data_tr))

    _, _, Dice = metric_val.get()
    print("Overall mean dice score is:", Dice)
    print("Finished test")

    dice_arr = 100 * np.reshape(dice_list, [opt.num_classes - 1, -1])

    dice_mean = np.mean(dice_arr, axis=1)
    dice_std = np.std(dice_arr, axis=1)

    print(dice_mean.shape, dice_arr.shape)
    print('Dice:')
    print('AA :%.2f(%.2f)' % (dice_mean[3], dice_std[3]))
    print('LAC:%.2f(%.2f)' % (dice_mean[1], dice_std[1]))
    print('LVC:%.2f(%.2f)' % (dice_mean[2], dice_std[2]))
    print('Myo:%.2f(%.2f)' % (dice_mean[0], dice_std[0]))
    print('Mean:%.2f' % np.mean(dice_mean))

    assd_arr = np.reshape(assd_list, [opt.num_classes - 1, -1])

    assd_mean = np.mean(assd_arr, axis=1)
    assd_std = np.std(assd_arr, axis=1)

    print('ASSD:')
    print('AA :%.2f(%.2f)' % (assd_mean[3], assd_std[3]))
    print('LAC:%.2f(%.2f)' % (assd_mean[1], assd_std[1]))
    print('LVC:%.2f(%.2f)' % (assd_mean[2], assd_std[2]))
    print('Myo:%.2f(%.2f)' % (assd_mean[0], assd_std[0]))
    print('Mean:%.2f' % np.mean(assd_mean))
    print("Finished test")
    print(ckpt_dir)



def inference(ckpt_dir, exp_name):
    # inference on a certain volumne and save the gd as well as the prediction mask and synthetic image

    save_dir = os.path.join("checkpoints/saved_imgs", exp_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    pred_dir = os.path.join(save_dir, "pred")
    synth_dir = os.path.join(save_dir, "synthesis")
    fake_pred_dir = os.path.join(save_dir, "fake_pred")
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    if not os.path.exists(synth_dir):
        os.mkdir(synth_dir)
    if not os.path.exists(fake_pred_dir):
        os.mkdir(fake_pred_dir)

    opt = TrainOptions().parse()
    split_dir = os.path.join(opt.target_dir, "splits.pkl")
    with open(split_dir, 'rb') as f:
        splits = pickle.load(f)

    test_keys = splits[opt.fold]['test']

    src_split_dir = os.path.join(opt.src_dir, "splits.pkl")
    with open(src_split_dir, 'rb') as f:
        src_splits = pickle.load(f)

    src_train_keys = src_splits[opt.fold]['train']
    src_train_keys.sort()

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # create model and oad state_dict
    model = create_model(opt)
    seg_model = model.netS_B.to(device)
    netF = model.netF.to(device)
    netG = model.netG.to(device)
    nce_layers = [0, 4, 8, 12, 16]

    state_dict = torch.load(os.path.join(ckpt_dir, "best_net_S_B.pth"))
    seg_model.load_state_dict(state_dict)
    state_dict = torch.load(os.path.join(ckpt_dir, "best_net_G.pth"))
    netG.load_state_dict(state_dict)

    metric_val = SegmentationMetric(opt.num_classes)

    with torch.no_grad():
        for k in test_keys:
            data_path = os.path.join(opt.target_dir, 'cropped', k)
            volume = np.load(data_path)[:, 0]
            label = np.load(data_path)[:, 1]

            slice_num = volume.shape[0]
            volume_pred = np.zeros(volume.shape)
            for i in range(0, slice_num):
                img = volume[i]
                img = torch.from_numpy(img[None, None]).float().to(device)
                output = seg_model(img)
                pred_softmax = F.softmax(output, dim=1)
                pred = torch.argmax(pred_softmax, dim=1).cpu().numpy()

                volume_pred[i] = pred.squeeze(0)

                metric_val.update(torch.from_numpy(label[i]).unsqueeze(0), pred_softmax)

            pred_file = os.path.join(pred_dir, 'pred_' + k)
            np.save(pred_file, volume_pred)

        index = 0
        for k in src_train_keys:
            data_path = os.path.join(opt.src_dir, 'cropped', k)
            volume = np.load(data_path)[:, 0]

            slice_num = volume.shape[0]
            volume_fake = np.zeros(volume.shape)
            volume_pred_fake = np.zeros(volume.shape)
            for i in range(0, slice_num):
                img = volume[i]
                img = torch.from_numpy(img[None, None]).float().to(device)
                fake_img = netG(img).cpu().numpy()
                output = seg_model(img)
                pred_softmax = F.softmax(output, dim=1)
                pred = torch.argmax(pred_softmax, dim=1).cpu().numpy()

                volume_pred_fake[i] = pred.squeeze(0)
                volume_fake[i] = fake_img.squeeze(0)

            synth_file = os.path.join(synth_dir, 'fake_' + k)
            fake_pred_file = os.path.join(fake_pred_dir, 'fp_' + k )

            np.save(synth_file, volume_fake)
            np.save(fake_pred_file, volume_pred_fake)

            if index > 5:
                break
            index += 1


if __name__ == "__main__":
    ckpt_dir = "checkpoints/cut_atten_coseg_sum_sifa_mr2ct_lr_gan_5e-5_e50_f0_b8_20211115-074216"
    evaluate(ckpt_dir)
    # inference(ckpt_dir, "cut_atten_coseg")
