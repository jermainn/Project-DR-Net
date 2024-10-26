
import torch.optim as optim
import torch
from model.DR_Net_hyper import DR_Net
from model.DR_Net_hyper import CONFIGS as CONFIGS_ViT

def CreateModel(args):
    
    config_vit = CONFIGS_ViT[args.vit_name]
    config_vit.n_skip = args.n_skip
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    model = DR_Net(config_vit, img_size=args.img_size, args=args)
    
    if args.restore_from is not None:
        model.load_state_dict(torch.load(args.restore_from, map_location=lambda storage, loc: storage))
    return model


if __name__ == '__main__':
    model = CreateModel()
    print(model)