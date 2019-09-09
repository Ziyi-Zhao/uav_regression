import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim import lr_scheduler
from uav_model import UAVModel


def train(model, train_loader, device, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    num_images = 0
    pass


def val(model, test_loader, device, criterion, epoch):
    model.eval()
    running_loss = 0.0
    pass


def save_model(checkpoint_dir, model_checkpoint_name, model):
    model_save_path = '{}/{}'.format(checkpoint_dir, model_checkpoint_name)
    print('save model to: \n{}'.format(model_save_path))
    torch.save(model, model_save_path)

def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="input data path", required=True, type=str)
    parser.add_argument("--lr", help="learning rate", required=True, type=float)
    parser.add_argument("--momentum", help="momentum", required=True, type=float)
    parser.add_argument("--weight_decay", help="weight decay", required=True, type=float)
    parser.add_argument("--batch_size", help="batch size", required=True, type=int)
    parser.add_argument("--num_epochs", help="num_epochs", required=True, type=int)
    parser.add_argument("--checkpoint_dir", help="checkpoint_dir", required=True, type=str)
    parser.add_argument("--model_checkpoint_name", help="model checkpoint name", required=True, type=str)
    parser.add_argument("--load_from_checkpoint", dest='load_from_checkpoint', action='store_true')
    parser.add_argument("--eval_only", dest='eval_only', action='store_true')
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    device = torch.device("cuda")
    print("\nLet's use", torch.cuda.device_count(), "GPUs!\n")
    model_ft = UAVModel()
    model_ft = nn.DataParallel(model_ft)

    if args.load_from_checkpoint:
        chkpt_model_path = os.path.join(args.checkpoint_dir, args.model_checkpoint_name)
        print("Loading ", chkpt_model_path)
        chkpt_model = torch.load(chkpt_model_path, map_location=device)
        model_ft.load_state_dict(chkpt_model.state_dict())

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Decay LR by a factor of 0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    model_ft = model_ft.to(device)

    best_loss = np.inf
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 80)
        exp_lr_scheduler.step()
        train()

    pass


if __name__ == '__main__':
    main()
