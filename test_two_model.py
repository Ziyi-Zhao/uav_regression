import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from model import mainnet
from seg_dynamic import seg_dynamic
from seg_static import seg_static
from dataloader_two_model import UAVDatasetTuple
from utils import visualize_sum_testing_result_cont
from correlation import Correlation
from auc import auc


image_saving_dir = '/home/zzhao/data/uav_regression/'


os.environ["CUDA_VISIBLE_DEVICES"]="1"

init_cor = Correlation()
pred_cor = Correlation()

def val_continuous(path, model_ft_dynamic, model_ft_static, test_loader, device, criterion, epoch, batch_size):
    model_ft_dynamic.eval()
    model_ft_static.eval()
    sum_running_loss = 0.0
    prediction_output_segment = []
    label_output_segment = []
    init_output_segment = []

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):

            task_label = data['task_label'].to(device).float()

            # All black
            init = data['init']
            init[:] = 0
            init = init.to(device).float()

            # Normal
            # init = data['init'].to(device).float()

            # print("init shape", init.shape)
            last_label = data['last_label'].to(device).float()
            avg_label = data['avg_label'].to(device).float()

            prediction_last = np.zeros(last_label[:, 1, :, :].shape)
            prediction_avg = np.zeros(avg_label[:, 1, :, :].shape)

            for i in range(avg_label.shape[1]):

                # model prediction
                if i == 0:
                    task_label_input = task_label[:, i, :, :, :]
                    init_input = init[:, i, :, :]
                    prediction_last = model_ft_dynamic(subx=task_label_input, mainx=init_input)
                    prediction_avg = model_ft_static(subx=task_label_input, mainx=init_input)
                else:
                    task_label_input = task_label[:, i, :, :, :]
                    prediction_last = prediction_last[:, None, :, :]
                    init_input = prediction_last

                    prediction_last = model_ft_dynamic(subx=task_label_input, mainx=init_input)
                    prediction_avg = model_ft_static(subx=task_label_input, mainx=init_input)
                # loss
                loss_mse = criterion(prediction_avg, avg_label[:, i, :, :].data)
                # print (loss_mse)

                # accumulate loss
                sum_running_loss += loss_mse.item() * init.size(0)

                # visualize the sum testing result
                visualize_sum_testing_result_cont(path, init_input, prediction_avg, task_label[:, i, :, :, :], avg_label[:, i, :, :].data,
                                             batch_idx, epoch, batch_size, i)

                prediction_temp = prediction_avg.cpu().detach().numpy()
                label_temp = avg_label[:, i, :, :].cpu().detach().numpy()
                init_temp = init[:, i, :, :].cpu().detach().numpy()

                # save all prediction, label, init results
                if batch_idx == 0 and i == 0:
                    prediction_output = prediction_temp
                    label_output = label_temp
                    init_output = init_temp
                else:
                    prediction_output = np.append(prediction_output, prediction_temp, axis=0)
                    label_output = np.append(label_output, label_temp, axis=0)
                    init_output = np.append(init_output, init_temp, axis=0)

                # save segment prediction, label, init results
                if batch_idx == 0:
                    prediction_output_segment.append(prediction_temp)
                    label_output_segment.append(label_temp)
                    init_output_segment.append(init_temp)
                else:
                    prediction_output_segment[i] = np.append(prediction_output_segment[i], prediction_temp, axis=0)
                    label_output_segment[i] = np.append(label_output_segment[i], label_temp, axis=0)
                    init_output_segment[i] = np.append(init_output_segment[i], init_temp, axis=0)

    sum_running_loss = sum_running_loss / (len(test_loader.dataset) * avg_label.shape[1])
    print('\nTesting phase: epoch: {} Loss: {:.4f}\n'.format(epoch, sum_running_loss))

    # save auroc result
    # auc_path = os.path.join(path, "epoch_" + str(epoch))
    # auc(['flow'], [2, 4, 10, 100], [[label_output, prediction_output]], auc_path)

    # save correlation result
    correlation_path = path
    cor_path = os.path.join(correlation_path, "epoch_" + str(epoch))
    correlation_pred_label = pred_cor.corrcoef(prediction_output, label_output, cor_path, "correlation_{0}.png".format(epoch))
    correlation_init_label = init_cor.corrcoef(init_output, label_output, cor_path,  "correlation_init_label_{0}.png".format(epoch))
    print('correlation coefficient : {0}\n'.format(correlation_pred_label))
    print('correlation_init_label coefficient : {0}\n'.format(correlation_init_label))

    for i in range(len(prediction_output_segment)):
        init_seg_cor = Correlation()
        pred_seg_cor = Correlation()
        correlation_pred_label = pred_seg_cor.corrcoef(prediction_output_segment[i], label_output_segment[i], cor_path,
                                 "correlation_{0}_{1}.png".format(epoch, i))
        correlation_init_label = init_seg_cor.corrcoef(init_output_segment[i], label_output_segment[i], cor_path,
                                                   "correlation_init_label_{0}_{1}.png".format(epoch, i))
        print('correlation coefficient segment {0} : {1}\n'.format(i, correlation_pred_label))
        print('correlation_init_label coefficient segment {0} : {1}\n'.format(i, correlation_init_label))
    return sum_running_loss, prediction_output, label_output, init_output


def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_label_path", help="data label path", required=True, type=str)
    parser.add_argument("--init_path", help="init path", required=True, type=str)
    parser.add_argument("--last_label_path", help="label path", required=True, type=str)
    parser.add_argument("--avg_label_path", help="label path", required=True, type=str)
    parser.add_argument("--batch_size", help="batch size", required=True, type=int)
    parser.add_argument("--split_ratio", help="training/testing split ratio", required=True, type=float)
    parser.add_argument("--load_from_last_checkpoint", type=str)
    parser.add_argument("--load_from_avg_checkpoint", type=str)
    parser.add_argument("--image_save_folder", type=str, required=True)
    parser.add_argument("--eval_only", dest='eval_only', action='store_true')
    args, unknown = parser.parse_known_args()

    image_saving_path = image_saving_dir + args.image_save_folder

    device = torch.device("cuda")

    all_dataset = UAVDatasetTuple(task_label_path = args.data_label_path,
                                  init_path=args.init_path,
                                  last_label_path=args.last_label_path,
                                  avg_label_path=args.avg_label_path)

    train_size = int(args.split_ratio * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    print("Total image tuples for train: ", len(train_dataset))
    print("Total image tuples for test: ", len(test_dataset))

    print("\nLet's use", torch.cuda.device_count(), "GPUs!\n")

    model_ft_dynamic = seg_dynamic()
    model_ft_static = seg_static()

    model_ft_dynamic = nn.DataParallel(model_ft_dynamic)
    model_ft_static = nn.DataParallel(model_ft_static)

    criterion  = nn.MSELoss(reduction='sum')

    if args.load_from_last_checkpoint:
        chkpt_last_model_path = args.load_from_last_checkpoint
        print("Loading ", chkpt_last_model_path)
        model_ft_dynamic.load_state_dict(torch.load(chkpt_last_model_path, map_location=device))

    if args.load_from_avg_checkpoint:
        chkpt_avg_model_path = args.load_from_avg_checkpoint
        print("Loading ", chkpt_avg_model_path)
        model_ft_static.load_state_dict(torch.load(chkpt_avg_model_path, map_location=device))


    model_ft_dynamic = model_ft_dynamic.to(device)
    model_ft_static = model_ft_static.to(device)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=30, drop_last=True)

    if args.eval_only:
        print("eval only")
        for epoch in range(1):
            # val(image_saving_path, model_ft, test_loader, device, criterion, epoch, args.batch_size)
            val_continuous(image_saving_path, model_ft_dynamic, model_ft_static, test_loader, device, criterion, epoch, args.batch_size)
        return True

if __name__ == '__main__':
    main()