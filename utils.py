import gc
import os
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
import logging
import numpy as np
from skimage import io
from torch.autograd import Variable
from basics import f1_mae_torch
import cv2
from tqdm import tqdm
from glob import glob
import pickle as pkl
from skimage.morphology import disk, skeletonize
from skimage.measure import label
import mlflow
import json
from models.isnet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model local
Path(os.path.sep.join([os.getcwd(), "weights"])).mkdir(parents=True, exist_ok=True)
weights_path = os.path.join(os.getcwd(), 'weights')


def train(net, optimizer, train_dataloaders, train_datasets, valid_dataloaders, valid_datasets,
          start_ite, model_digit, early_stop, model_save_fre,
          batch_size_train, batch_size_valid, max_ite, max_epoch_num):
    model_path = weights_path
    model_save_fre = model_save_fre
    max_ite = max_ite

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    ite_num = start_ite  # count the total iteration number
    ite_num4val = 0  #
    running_loss = 0.0  # count the toal loss
    running_tar_loss = 0.0  # count the target output loss
    last_f1 = [0 for x in range(len(valid_dataloaders))]

    train_num = train_datasets [0].__len__()

    net.train()

    start_last = time.time()
    gos_dataloader = train_dataloaders [0]
    epoch_num = max_epoch_num
    notgood_cnt = 0
    for epoch in range(epoch_num):  # set the epoch num as 100000

        for i, data in enumerate(gos_dataloader):

            if ite_num >= max_ite:
                print("Training Reached the Maximal Iteration Number ", max_ite)

            # start_read = time.time()
            ite_num = ite_num + 1
            ite_num4val = ite_num4val + 1

            # get the inputs
            inputs, labels = data ['image'], data ['label']

            if model_digit == "full":
                inputs = inputs.type(torch.FloatTensor)
                labels = labels.type(torch.FloatTensor)
            else:
                inputs = inputs.type(torch.HalfTensor)
                labels = labels.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                            requires_grad=False)
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

            # y zero the parameter gradients
            start_inf_loss_back = time.time()
            optimizer.zero_grad()

            # forward + backward + optimize
            ds, _ = net(inputs_v)
            # loss2, loss = net.compute_loss(ds, labels_v)
            loss2, loss = muti_loss_fusion(ds, labels_v)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()

            # del outputs, loss
            del ds, loss2, loss
            end_inf_loss_back = time.time() - start_inf_loss_back

            print(">>>" + model_path.split('/') [
                -1] + " - [epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f, time-per-iter: %3f s, time_read: %3f" % (
                      epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
                      running_tar_loss / ite_num4val, time.time() - start_last,
                      time.time() - start_last - end_inf_loss_back))

            start_last = time.time()

            if ite_num % model_save_fre == 0:  # e.g validate every 2000 iterations
                notgood_cnt += 1
                net.eval()
                tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time, tmp_prec, tmp_rec, tmp_iou = valid(net,
                                                                                                         valid_dataloaders,
                                                                                                         valid_datasets,
                                                                                                         model_digit,
                                                                                                         batch_size_valid,
                                                                                                         max_epoch_num,
                                                                                                         epoch)
                net.train()  # resume train

                tmp_out = 0
                print("last_f1:", last_f1)
                print("tmp_f1:", tmp_f1)
                for fi in range(len(last_f1)):
                    if (tmp_f1 [fi] > last_f1 [fi]):
                        tmp_out = 1
                print("tmp_out:", tmp_out)

                if tmp_out:
                    notgood_cnt = 0
                    last_f1 = tmp_f1
                    tmp_f1_str = [str(round(f1x, 4)) for f1x in tmp_f1]
                    tmp_mae_str = [str(round(mx, 4)) for mx in tmp_mae]
                    maxf1 = '_'.join(tmp_f1_str)
                    meanM = '_'.join(tmp_mae_str)
                    # .cpu().detach().numpy()
                    model_name = "/gpu_itr_" + str(ite_num) + \
                                 "_traLoss_" + str(np.round(running_loss / ite_num4val, 4)) + \
                                 "_traTarLoss_" + str(np.round(running_tar_loss / ite_num4val, 4)) + \
                                 "_valLoss_" + str(np.round(val_loss / (i_val + 1), 4)) + \
                                 "_valTarLoss_" + str(np.round(tar_loss / (i_val + 1), 4)) + \
                                 "_maxF1_" + maxf1 + \
                                 "_mae_" + meanM + \
                                 "_time_" + str(np.round(np.mean(np.array(tmp_time)) / batch_size_valid, 6)) + ".pth"
                    torch.save(net.state_dict(), model_path + model_name)
                # mlflow
                mlflow.log_metric("train_loss", np.round(running_loss / ite_num4val, 4), step=ite_num)
                mlflow.log_metric("val_loss", np.round(val_loss / (i_val + 1), 4), step=ite_num)
                mlflow.log_metric("F1_score", round(tmp_f1 [0], 4), step=ite_num)
                mlflow.log_metric("mae", round(tmp_mae [0], 4), step=ite_num)
                mlflow.log_metric("prec", round(tmp_prec [0], 4), step=ite_num)
                mlflow.log_metric("rec", round(tmp_rec [0], 4), step=ite_num)
                mlflow.log_metric("iou", round(tmp_iou [0], 4), step=ite_num)

                running_loss = 0.0
                running_tar_loss = 0.0
                ite_num4val = 0

                if notgood_cnt >= early_stop:
                    print("No improvements in the last " + str(
                        notgood_cnt) + " validation periods, so training stopped !")
                    logging.info("No improvements in the last " + str(
                        notgood_cnt) + " validation periods, so training stopped !")
                    return

    print("Training Reaches The Maximum Epoch Number")


def valid(net, valid_dataloaders, valid_datasets, model_digit, batch_size_valid, max_epoch_num, epoch=0):
    net.eval()
    print("Validating...")
    epoch_num = max_epoch_num

    val_loss = 0.0
    tar_loss = 0.0
    val_cnt = 0.0

    tmp_f1 = []
    tmp_mae = []
    tmp_prec = []
    tmp_rec = []
    tmp_iou = []
    tmp_time = []

    start_valid = time.time()

    for k in range(len(valid_dataloaders)):

        valid_dataloader = valid_dataloaders [k]
        valid_dataset = valid_datasets [k]

        val_num = valid_dataset.__len__()
        mybins = np.arange(0, 256)
        PRE = np.zeros((val_num, len(mybins) - 1))
        REC = np.zeros((val_num, len(mybins) - 1))
        F1 = np.zeros((val_num, len(mybins) - 1))
        MAE = np.zeros((val_num))
        IOU = np.zeros((val_num))

        for i_val, data_val in enumerate(valid_dataloader):
            val_cnt = val_cnt + 1.0
            imidx_val, inputs_val, labels_val, shapes_val = data_val ['imidx'], data_val ['image'], data_val ['label'], \
                data_val ['shape']

            if model_digit == "full":
                inputs_val = inputs_val.type(torch.FloatTensor)
                labels_val = labels_val.type(torch.FloatTensor)
            else:
                inputs_val = inputs_val.type(torch.HalfTensor)
                labels_val = labels_val.type(torch.HalfTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_val_v, labels_val_v = Variable(inputs_val.cuda(), requires_grad=False), Variable(
                    labels_val.cuda(), requires_grad=False)
            else:
                inputs_val_v, labels_val_v = Variable(inputs_val, requires_grad=False), Variable(labels_val,
                                                                                                 requires_grad=False)

            t_start = time.time()
            ds_val = net(inputs_val_v) [0]
            t_end = time.time() - t_start
            tmp_time.append(t_end)

            # loss2_val, loss_val = net.compute_loss(ds_val, labels_val_v)
            loss2_val, loss_val = muti_loss_fusion(ds_val, labels_val_v)

            # compute F measure
            for t in range(batch_size_valid):
                i_test = imidx_val [t].data.numpy()

                pred_val = ds_val [0] [t, :, :, :]  # B x 1 x H x W

                # recover the prediction spatial size to the orignal image size
                pred_val = torch.squeeze(
                    F.upsample(torch.unsqueeze(pred_val, 0), (shapes_val [t] [0], shapes_val [t] [1]), mode='bilinear'))

                # pred_val = normPRED(pred_val)
                ma = torch.max(pred_val)
                mi = torch.min(pred_val)
                pred_val = (pred_val - mi) / (ma - mi)  # max = 1

                if len(valid_dataset.dataset ["ori_gt_path"]) != 0:
                    gt = np.squeeze(io.imread(valid_dataset.dataset ["ori_gt_path"] [i_test]))  # max = 255
                else:
                    gt = np.zeros((shapes_val [t] [0], shapes_val [t] [1]))
                with torch.no_grad():
                    gt = torch.tensor(gt).to(device)

                pre, rec, f1, mae, iou = f1_mae_torch(pred_val * 255, gt, valid_dataset, i_test, mybins)

                PRE [i_test, :] = pre
                REC [i_test, :] = rec
                F1 [i_test, :] = f1
                MAE [i_test] = mae
                IOU [i_test] = iou

                del ds_val, gt
                gc.collect()
                torch.cuda.empty_cache()

            # if(loss_val.data[0]>1):
            val_loss += loss_val.item()  # data[0]
            tar_loss += loss2_val.item()  # data[0]

            print("[validating: %5d/%5d] val_ls:%f, tar_ls: %f, f1: %f, mae: %f, iou: %f, time: %f" % (
                i_val, val_num, val_loss / (i_val + 1), tar_loss / (i_val + 1), np.amax(F1 [i_test, :]), MAE [i_test],
                IOU [i_test], t_end))

            del loss2_val, loss_val

        print('============================')
        PRE_m = np.mean(PRE, 0)
        REC_m = np.mean(REC, 0)
        f1_m = (1 + 0.3) * PRE_m * REC_m / (0.3 * PRE_m + REC_m + 1e-8)

        tmp_f1.append(np.amax(f1_m))
        tmp_mae.append(np.mean(MAE))
        tmp_prec.append(np.mean(PRE_m))
        tmp_rec.append(np.mean(REC_m))
        tmp_iou.append(np.mean(IOU))

    return tmp_f1, tmp_mae, val_loss, tar_loss, i_val, tmp_time, tmp_prec, tmp_rec, tmp_iou


def muti_loss_fusion(preds, target):
    bce_loss = nn.BCELoss(size_average=True)
    loss0 = 0.0
    loss = 0.0

    for i in range(0, len(preds)):
        # print("i: ", i, preds[i].shape)
        if (preds [i].shape [2] != target.shape [2] or preds [i].shape [3] != target.shape [3]):
            # tmp_target = _upsample_like(target,preds[i])
            tmp_target = F.interpolate(target, size=preds [i].size() [2:], mode='bilinear', align_corners=True)
            loss = loss + bce_loss(preds [i], tmp_target)
        else:
            loss = loss + bce_loss(preds [i], target)
        if (i == 0):
            loss0 = loss
    return loss0, loss


def create_inputs(train_dir, val_dir):
    dataset_train = {"name": 'TRAIN_DATA',
                     "im_dir": os.path.join(train_dir, 'im'),
                     "gt_dir": os.path.join(train_dir, 'gt'),
                     "im_ext": ".jpg",
                     "gt_ext": ".png",
                     "cache_dir": os.path.join(train_dir, 'cache')}

    dataset_val = {"name": 'VAL_DATA',
                   "im_dir": os.path.join(val_dir, 'im'),
                   "gt_dir": os.path.join(val_dir, 'gt'),
                   "im_ext": ".jpg",
                   "gt_ext": ".png",
                   "cache_dir": os.path.join(val_dir, 'cache')}

    train_datasets = [dataset_train]
    valid_datasets = [dataset_val]

    return train_datasets, valid_datasets


def dataset_info(image_dir, mask_dir):
    dataset_info = {}

    image_paths = []
    mask_paths = []

    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))

    for root, _, files in os.walk(mask_dir):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                mask_paths.append(os.path.join(root, file))

    dataset_info ['num_images'] = len(image_paths)
    dataset_info ['im_paths'] = image_paths
    dataset_info ['gt_paths'] = mask_paths

    return dataset_info


def save_json_file(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)


