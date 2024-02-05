from matplotlib import pyplot as plt

from alg2_low_light_B.ENHANCENET import ENHANCENET
import argparse
from alg2_low_light_B.utils import *

# set the configuration to run the alg2_low_light_B.
def parse_args(path_for_processing):
    desc = "Pytorch implementation of NightImageEnhancement"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='[train / test]')
    parser.add_argument('--epoch', type=int, default=1, help='[train / test]')
    parser.add_argument('--dataset', type=str, default=path_for_processing, help='dataset_name')
    parser.add_argument('--datasetpath', type=str, default=path_for_processing, help='dataset_path')
    parser.add_argument('--iteration', type=int, default=900000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=100000, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='The weight decay')
    parser.add_argument('--atten_weight', type=int, default=0.5, help='Weight for Attention Loss')
    parser.add_argument('--use_gray_feat_loss', type=str2bool, default=True,
                        help='use Structure and HF-Features Consistency Losses')
    parser.add_argument('--feat_weight', type=int, default=1,
                        help='Weight for Structure and HF-Features Consistency Losses')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight for GAN Loss')
    parser.add_argument('--identity_weight', type=int, default=5, help='Weight for Identity Loss')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=512, help='The training size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default="LOL", help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=True)

    parser.add_argument('--im_suf_A', type=str, default='.jpg', help='The suffix of test images [.png / .jpg]')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    check_folder("alg2_low_light_B/results/LOL/model/")
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


def show(img):
    img = cv2.convertScaleAbs(img)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


"""main"""


def run_lowlight_B(frame):
    print("Running low light enhancement using Deep Learning Method.")

    args = parse_args("test_video")
    if args is None:
        exit()
    frame_lists_for_processing = [frame]


    # initialize model
    gan = ENHANCENET(args, frame_lists_for_processing)
    gan.build_model()

    if args.phase == 'test':
        # test the result
        result = gan.test()
        print(" Test finished!")
        return result[0]
