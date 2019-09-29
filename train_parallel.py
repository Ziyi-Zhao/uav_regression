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


def train(model, train_loader, device, structure, optimizer, criterion, weight, epoch):
    model.train()
    sum_running_loss = 0.0
    loss = 0.0
    num_images = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        # zero the parameter gradients
        optimizer.zero_grad()

        if structure == 'pnet':
            image = data['data'].to(device)
            density = data['density'].to(device)
            label_sum = data['label'].to(device)
            # model prediction
            prediction = model(image)
            loss_mean_square_error = criterion(prediction, label_sum.data)
            # loss
            loss = loss_mean_square_error

            # update the weights within the model
            loss.backward()
            optimizer.step()

            # accumulate loss
            if loss_mean_square_error != 0.0:
                sum_running_loss += loss_mean_square_error * image.size(0)
            num_images += image.size(0)

            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                sum_epoch_loss = sum_running_loss / num_images
                print('\nTraining phase: epoch: {} batch:{} Loss: {:.4f}\n'.format(epoch, batch_idx, sum_epoch_loss))
        elif structure == 'rnet':
            image = data['data'].to(device)
            label_sum = data['label'].to(device)
            # model prediction
            prediction = model(image)
            loss_mean_square_error = criterion(prediction, label_sum.data)
            # final loss
            loss = loss_mean_square_error

            # update the weights within the model
            loss.backward()
            optimizer.step()

            # accumulate loss
            sum_running_loss += loss_mean_square_error.item() * image.size(0)
            num_images += image.size(0)
            if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
                sum_epoch_loss = sum_running_loss / num_images
                print('\nTraining phase: epoch: {} batch:{} Loss: {:.4f}\n'.format(epoch, batch_idx, sum_epoch_loss))


def val(model, test_loader, device, structure, criterion, weight, epoch):
    model.eval()
    sum_running_loss = 0.0
    num_images = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            if structure == 'pnet':
                image = data['data'].to(device)
                density = data['density'].to(device)
                label_sum = data['label'].to(device)
                # model prediction
                prediction = model(image)
                loss_mean_square_error = criterion(prediction, label_sum.data)

                # accumulate loss
                sum_running_loss += loss_mean_square_error.item() * image.size(0)
                num_images += image.size(0)

                # visualize the sum testing result
                visualize_sum_testing_result(prediction, label_sum.data, batch_idx, epoch)
            elif structure == 'rnet':
                image = data['data'].to(device)
                label_sum = data['label'].to(device)
                # model prediction
                prediction = model(image)

                loss_mean_square_error = criterion(prediction, label_sum.data)

                # accumulate loss
                sum_running_loss += loss_mean_square_error.item() * image.size(0)
                num_images += image.size(0)

                # visualize the sum testing result
                visualize_sum_testing_result(prediction, label_sum.data, batch_idx, epoch)

        if structure == 'pnet':
            sum_test_loss = sum_running_loss / len(test_loader.dataset)
            loss = sum_test_loss
            print('\nTesting phase: epoch: {} Loss: {:.4f}\n'.format(epoch, sum_test_loss))
        elif structure == 'rnet':
            sum_test_loss = sum_running_loss / len(test_loader.dataset)
            loss = sum_test_loss
            print('\nTesting phase: epoch: {} SUM Loss: {:.4f}\n'.format(epoch, sum_test_loss))
    return loss


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

    all_dataset = UAVDatasetTuple(image_path=args.data_path, mode="train", structure=args.structure)
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

    # Define criterion
    criterion = nn.MSELoss(reduction='sum')

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=30)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=30)

    if args.eval_only:
        loss = val(model_ft, test_loader, device, args.structure, criterion, weight, 0)
        print('\nTesting phase: epoch: {} Loss: {:.4f}\n'.format(0, loss))
        return True

    best_loss = np.inf
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 80)
        exp_lr_scheduler.step()
        train(model_ft, train_loader, device, args.structure, optimizer_ft, criterion, weight, epoch)
        loss = val(model_ft, test_loader, device, args.structure, criterion, weight, epoch)
        if loss < best_loss:
            save_model(checkpoint_dir=args.checkpoint_dir,
                       model_checkpoint_name=args.model_checkpoint_name + '_' + str(loss),
                       model=model_ft)
            best_loss = loss


if __name__ == '__main__':
    main()
