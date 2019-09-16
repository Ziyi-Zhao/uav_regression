import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim import lr_scheduler
from uav_model import UAVModel
from data_loader import UAVDatasetTuple
from utils import draw_roc_curve, calculate_precision_recall, visualize_sum_testing_result, visualize_lstm_testing_result


os.environ["CUDA_VISIBLE_DEVICES"]="1"


def train(model, train_loader, device, optimizer, criterion_lstm, criterion_sum, weight, epoch):
    model.train()
    lstm_running_loss = 0.0
    sum_running_loss = 0.0
    num_images = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        image = data['data'].to(device)
        label_lstm = data['label_lstm'].to(device)
        label_sum = data['label_sum'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # model prediction
        lstm_prediction, sum_prediction = model(image)
        # lstm_prediction = model(image)
        loss_binary_cross_entropy = criterion_lstm(lstm_prediction, label_lstm.data)
        weight_ = weight[label_lstm.data.view(-1).long()].view_as(label_lstm)
        loss_binary_cross_entropy_weighted = loss_binary_cross_entropy * weight_.to(device)
        loss_binary_cross_entropy_weighted = loss_binary_cross_entropy_weighted.mean()
        loss_mean_square_error = criterion_sum(sum_prediction, label_sum.data)

        # combine the two way loss
        loss = loss_binary_cross_entropy_weighted + loss_mean_square_error

        # update the weights within the model
        loss.backward()
        optimizer.step()

        # accumulate loss
        lstm_running_loss += loss_binary_cross_entropy_weighted.item() * image.size(0)
        sum_running_loss += loss_mean_square_error * image.size(0)
        num_images += image.size(0)

        if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
            lstm_epoch_loss = lstm_running_loss / num_images
            sum_epoch_loss = sum_running_loss / num_images
            lstm_prediction_np, label_lstm_np = np.array(lstm_prediction.cpu().detach()), np.array(label_lstm.cpu().detach())
            precision, recall = calculate_precision_recall(lstm_prediction_np, label_lstm_np, "train", batch_idx, epoch)
            auroc = draw_roc_curve(lstm_prediction_np, label_lstm_np, "train", epoch, batch_idx)
            print('\nTraining phase: epoch: {} batch:{} LSTM Loss: {:.4f} SUM Loss: {:.4f} Precision: {:.4f} Recall: {:.4f} AUROC: {:.4f}\n'.format(epoch, batch_idx, lstm_epoch_loss, sum_epoch_loss, precision, recall, auroc))


def val(model, test_loader, device, criterion_lstm, criterion_sum, weight, epoch):
    model.eval()
    lstm_running_loss = 0.0
    sum_running_loss = 0.0
    num_images = 0
    precision = 0.0
    recall = 0.0
    loss_mean_square_error = 0.0
    lstm_prediction = None
    label_lstm = None

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            image = data['data'].to(device)
            label_lstm = data['label_lstm'].to(device)
            label_sum = data['label_sum'].to(device)

            # model prediction
            lstm_prediction, sum_prediction = model(image)
            # lstm_prediction = model(image)
            loss_binary_cross_entropy = criterion_lstm(lstm_prediction, label_lstm.data)
            weight_ = weight[label_lstm.data.view(-1).long()].view_as(label_lstm)
            loss_binary_cross_entropy_weighted = loss_binary_cross_entropy * weight_.to(device)
            loss_binary_cross_entropy_weighted = loss_binary_cross_entropy_weighted.mean()
            loss_mean_square_error = criterion_sum(sum_prediction, label_sum.data)

            # combine the two way loss
            loss = loss_binary_cross_entropy_weighted + loss_mean_square_error

            # accumulate loss
            lstm_running_loss += loss_binary_cross_entropy_weighted.item() * image.size(0)
            sum_running_loss += loss_mean_square_error.item() * image.size(0)
            num_images += image.size(0)

            # visualize the lstm testing result
            visualize_lstm_testing_result(lstm_prediction, label_lstm.data, batch_idx, epoch)

            # visualize the sum testing result
            visualize_sum_testing_result(sum_prediction, label_sum.data, batch_idx, epoch)

    lstm_test_loss = lstm_running_loss / len(test_loader.dataset)
    sum_test_loss = sum_running_loss / len(test_loader.dataset)

    lstm_prediction_np, label_lstm_np = np.array(lstm_prediction.cpu().detach()), np.array(label_lstm.cpu().detach())
    precision, recall = calculate_precision_recall(lstm_prediction_np, label_lstm_np, "test", batch_idx, epoch)
    auroc = draw_roc_curve(lstm_prediction_np, label_lstm_np, "test", epoch, 0)
    print('\nTesting phase: epoch: {} LSTM Loss: {:.4f} SUM Loss: {:.4f} Precision: {:.4f} Recall: {:.4f} AUROC: {:.4f}\n'.format(epoch, lstm_test_loss, sum_test_loss, precision, recall, auroc))

    return loss_mean_square_error, recall


def save_model(checkpoint_dir, model_checkpoint_name, model):
    model_save_path = '{}/{}'.format(checkpoint_dir, model_checkpoint_name)
    print('save model to: \n{}'.format(model_save_path))
    torch.save(model.state_dict(), model_save_path)


def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="data path", required=True, type=str)
    parser.add_argument("--structure", help="the structure of the feature embedding model", required=True, type=str)
    parser.add_argument("--lr", help="learning rate", required=True, type=float)
    parser.add_argument("--momentum", help="momentum", required=True, type=float)
    parser.add_argument("--weight_decay", help="weight decay", required=True, type=float)
    parser.add_argument("--batch_size", help="batch size", required=True, type=int)
    parser.add_argument("--num_epochs", help="num_epochs", required=True, type=int)
    parser.add_argument("--split_ratio", help="training/testing split ratio", required=True, type=float)
    parser.add_argument("--checkpoint_dir", help="checkpoint_dir", required=True, type=str)
    parser.add_argument("--model_checkpoint_name", help="model checkpoint name", required=True, type=str)
    parser.add_argument("--load_from_checkpoint", dest='load_from_checkpoint', action='store_true')
    parser.add_argument("--eval_only", dest='eval_only', action='store_true')
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    device = torch.device("cuda")

    all_dataset = UAVDatasetTuple(image_path=args.data_path, mode="train")
    positive_ratio, negative_ratio = all_dataset.get_class_count()
    weight = torch.FloatTensor((positive_ratio, negative_ratio))
    train_size = int(args.split_ratio * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    print("Total image tuples for train: ", len(train_dataset))
    print("Total image tuples for test: ", len(test_dataset))

    print("\nLet's use", torch.cuda.device_count(), "GPUs!\n")
    model_ft = UAVModel(args.structure)
    model_ft = nn.DataParallel(model_ft)

    criterion_lstm = nn.BCELoss(reduce=False)
    criterion_sum = nn.MSELoss()

    if args.load_from_checkpoint:
        chkpt_model_path = os.path.join(args.checkpoint_dir, args.model_checkpoint_name)
        print("Loading ", chkpt_model_path)
        chkpt_model = torch.load(chkpt_model_path, map_location=device)
        model_ft.load_state_dict(chkpt_model)
        model_ft.eval()

    model_ft = model_ft.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Decay LR by a factor of 0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=30)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=30)

    if args.eval_only:
        loss = val(model_ft, test_loader, device, criterion_lstm, criterion_sum, weight, 0)
        print('\nTesting phase: epoch: {} Loss: {:.4f}\n'.format(0, loss))
        return True

    best_loss = np.inf
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 80)
        exp_lr_scheduler.step()
        train(model_ft, train_loader, device, optimizer_ft, criterion_lstm, criterion_sum, weight, epoch)
        loss_mean_square_error, recall = val(model_ft, test_loader, device, criterion_lstm, criterion_sum, weight, epoch)
        if loss_mean_square_error < best_loss:
            # save_model(checkpoint_dir=args.checkpoint_dir,
            #            model_checkpoint_name=args.model_checkpoint_name +
            #                                  str(loss_mean_square_error.cpu().detach()).replace('(', '_').replace(')', ''),
            #            model=model_ft)
            best_loss = loss_mean_square_error


if __name__ == '__main__':
    main()
