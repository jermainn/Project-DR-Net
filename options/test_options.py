import argparse
import os.path as osp

class TestOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="DR-Net Model")
        # basic parameters
        parser.add_argument('--is-train', default=False, type=bool, help='Training')
        parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
        parser.add_argument('--num-workers', default=8, type=int, help='number of data loading workers (default: 4)')
        parser.add_argument('--snapshot-dir', type=str, default='./experiments/', help='where to save models')
        parser.add_argument('--restore-from', type=str, default=r"./pretrained_model/test_model.pth", help='where to restore models from')
        # dataset parameters
        parser.add_argument('--dataroot', default='./data', help='path to images')
        # ViT config
        parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
        parser.add_argument('--seed', type=int,
                            default=1314, help='random seed')
        parser.add_argument('--n_skip', type=int,
                            default=3, help='using number of skip-connect, default is num')
        parser.add_argument('--vit_name', type=str,
                            default='R50-ViT-B_16', help='select one vit model')
        parser.add_argument('--vit_patches_size', type=int,
                            default=16, help='vit_patches_size, default is 16')
        # ir-gan
        parser.add_argument('--ir-gan-downsample', default=4, type=int,
                                help='number of downsampling layers in ir-gan')
        parser.add_argument('--no-ir-gan-skip', dest='ir_gan_skip', action='store_false', help='dont use u-net skip connections in the ir-gan')
        parser.add_argument('--n-res', default=8, type=int, help='number of residual blocks')
        parser.add_argument('--norm', default='gn', type=str, help='type of normalization layer')
        parser.add_argument('--denormalize', dest='denormalize', action='store_true', help='denormalize output image by input image mean/var')
        parser.add_argument('--ir-gan-dys', type=bool, default=True, help='use DySample in ir-gan')
        parser.add_argument('--ir-gan-cbam-skip', type=bool, default=True, help='use cbam_skip in net')
        parser.add_argument('--dim', default=32, type=int, help='initial feature dimension (doubled at each downsampling layer)')
        # DE-Net
        parser.add_argument('--freeze-de-net', dest='freeze_de_net', default=False, help='dont train the de net')
        # IR-GAN
        # parser.add_argument('--Z-adv', default=0, type=float, help='weight of adversarial loss after color net')
        parser.add_argument('--ir-gan', dest='ir_gan', default=True, help='do not include ir-gan in the model')
        # parser.add_argument()

        return parser.parse_args()


