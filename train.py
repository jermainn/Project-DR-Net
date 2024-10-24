from options.train_options import TrainOptions
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
        ]))
    
    print("The length of train set is: {}".format(len(train_data)))

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, 
        num_workers=args.num_workers, pin_memory=True, 
        shuffle=True)

    print("fixed test batch for visualization during training")
    fixed_batch = iter(test_loader).next()[0]
    
    return train_loader, test_loader, fixed_batch

def log_training_info(writer, lr_, iter_num, epoch_num, losses, image_batch, target_batch, pred):
    writer.add_scalar('info/lr', lr_, iter_num)
    for key in losses.keys():
        writer.add_scalar(f'info/{key}', losses[key], iter_num)
    logging.info('iteration %d epoch: %d: lr: %f, z_con: %f, z_per: %f, z_adv: %f, l_dic: %f, l_con: %f, l_per: %f, x_l1: %f'
                % (iter_num, (epoch_num+1), lr_,
                losses['Z_con'].item(), losses['Z_per'].item(), losses['Z_adv'].item(), losses['L_dic'].item(), 
                losses['L_con'].item(), losses['L_per'].item(), losses['X_L1'].item()))
    image = (image_batch[0, :, :, :] + 1.0) / 2.0
    writer.add_image('train/Image', image, iter_num)
    output = (pred[0, :, :, :] + 1.0) / 2.0
    writer.add_image('train/Prediction', output, iter_num)
    target = (target_batch[0, :, :, :] + 1.0) / 2.0
    writer.add_image('train/GroundTruth', target, iter_num)


def main():
    opt = TrainOptions()
    args = opt.initialize()

    # basic setup
    # model_name = args.model_name
    model_name = 'DR-Net'
    if model_name is not None:
        args.snapshot_dir = os.path.join(args.snapshot_dir, model_name)
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
        os.makedirs(os.path.join(args.snapshot_dir, 'log'))
    opt.print_options(args)
    
    logging.basicConfig(filename=args.snapshot_dir + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    # train writer
    writer = SummaryWriter(
        os.path.join(args.snapshot_dir, "log", model_name if model_name is not None else 'default')
    )

    # dataloader and model
    train_loader, _, _ = create_dataloader(args)
    model = CreateModel(args)
    # print(model)
    # exit()
    from torchsummary import summary
    summary(model, input_data=(3, 224, 224))
    # exit()
    # cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    model.train()
    
    start_epoch = 0
    if args.restore_from is not None:
        start_epoch = int(args.restore_from.rsplit('.', 1)[0].rsplit('/', 1)[1].rsplit('_', 1)[1])
    max_iter = args.max_epochs * len(train_loader)
    iter_num = 0
    logging.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iter))

    _t = {'iter time' : Timer()}
    _t['iter time'].tic()
    iterator = tqdm(range(start_epoch, args.max_epochs))
    for epoch_num in iterator:
        epoch_iters = 0
        for i_batch, ((image_batch, target_batch), _) in enumerate(train_loader):
            # optimizer.zero_grad()
            image_batch = image_batch.cuda()
            target_batch = target_batch.cuda() 

            # predict
            pred, losses = model.optimize_parameters(image_batch, target_batch)
            _t['iter time'].toc(average=False)
            
            # log 
            if iter_num % 200 == 0:
                for param_group in model.optimizer_G.param_groups:
                    lr_ = param_group['lr']
                log_training_info(writer, lr_, iter_num, epoch_num, losses, image_batch, target_batch, pred)
            
            iter_num += 1
            epoch_iters += args.batch_size

            _t['iter time'].tic()
        
        if not (epoch_num+1) % args.save_pred_epoch:
            save_mode_path = os.path.join(args.snapshot_dir, 'epoch_' + str(epoch_num+1) + '.pth')
            # save_optimizer_path = os.path.join(args.snapshot_dir, 'optimizer_epoch_' + str(epoch_num+1) + '.pth')
            torch.save(model.state_dict(), save_mode_path, _use_new_zipfile_serialization=False)
            logging.info("save model to {}".format(save_mode_path))
            
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    main()
