from . import common_args, modify_args
from common.util import str2bool

def get_args():
    parser = common_args.get_args()
    parser.add_argument('--srnet_pth', default='../pretrained/edsr_lambda0.5.pth', type=str, help='')
    parser.add_argument('--down_factor', default=1, type=int, help='downsample factor')
    parser.add_argument('--size', default=-1, type=int, help='downsample to this size (-1: use down_factor)')
    parser.add_argument('--isSR', default=True, type=str2bool, help='use EDSR?')
    args = modify_args.run(parser)
    return args