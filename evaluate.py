from options.test_options import TestOptions
import os
import sys
import logging
from tqdm import tqdm
from torch import optim
from model import CreateModel
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils.timer import Timer
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from dataset.datasets_water import ImageFolder, PairedImageFolder
from dataset import pairedtransforms
from torchvision import transforms
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def create_dataloader(args):
    # dataset path
    train_dir_1 = os.path.join(args.dataroot, 'train', 'input')
    train_dir_2 = os.path.join(args.dataroot, 'train', 'truth')
    val_dir_1 = os.path.join(args.dataroot, 'test', 'input')
    val_dir_2 = os.path.join(args.dataroot, 'test', 'truth')

    train_data = PairedImageFolder(
        train_dir_1,
        train_dir_2,
        transform=transforms.Compose([
            pairedtransforms.RandomResizedCrop(args.img_size),
            pairedtransforms.RandomHorizontalFlip(),
            pairedtransforms.ToTensor(),
            pairedtransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    )

    test_data = PairedImageFolder(val_dir_1, val_dir_2, transform=
        transforms.Compose([
            pairedtransforms.Resize(args.img_size),
            pairedtransforms.CenterCrop(args.img_size),
            pairedtransforms.ToTensor(),
            pairedtransforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]), return_path=True)
    
    # print("The length of train set is: {}".format(len(train_data)))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=True, 
        shuffle=True)
    # print("fixed test batch for visualization during training")
    fixed_batch = iter(test_loader).next()[0]
    
    return train_loader, test_loader, fixed_batch


def main():
    opt = TestOptions()
    args = opt.initialize()

    # basic setup
    # model_name = args.model_name
    model_name = 'DR-Net'
    if model_name is not None:
        args.snapshot_dir = os.path.join(args.snapshot_dir, model_name)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'results'))

    # dataloader and model
    _, test_loader, _ = create_dataloader(args)
    model = CreateModel(args)
    # cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for i_batch, ((image_batch, target_batch), _, names) in tqdm(enumerate(test_loader), total=len(test_loader)):
            image_batch = image_batch.cuda()
            target_batch = target_batch.cuda()
            _, _, _, z = model(image_batch)

            # save the output image
            for out, name in zip(z, names):
                save_path = os.path.join(args.snapshot_dir, "results", os.path.dirname(name))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                im = Image.fromarray((out * .5 + .5).mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy())
                im.save(os.path.join(save_path, os.path.basename(name)))

    return "Process Finished!"


if __name__ == "__main__":
    main()
