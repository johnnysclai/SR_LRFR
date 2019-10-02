import argparse
from . import common_args, modify_args
from common.util import str2bool


def get_args():
	parser = common_args.get_args()
	parser.add_argument('--isTrain', default=True, type=str2bool, help='is train?')
	## Training settings
	parser.add_argument('--dataset', default='celeba', type=str,
	                    help='webface/vggface2/celeba')
	parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
	parser.add_argument('--bs', default=64, type=int, help='')
	parser.add_argument('--iterations', default=50000, type=int, help='50000')
	parser.add_argument('--decay_step', default=25000, type=int, help='25000')
	## SR model settings
	parser.add_argument('--sr_net', default='EDSR', type=str, help='EDSR')
	parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
	                    help='super resolution upscale factor')
	parser.add_argument('--num_resblocks', default=32, type=int, help='')
	parser.add_argument('--num_filters', default=256, type=int, help='')
	parser.add_argument('--lamb_id', default=0.5, type=float, help='lambda')
	## Other settings
	parser.add_argument('--name', type=str, default='master',
	                    help='name of the experiment. It decides where to store samples and models')
	parser.add_argument('--checkpoints_dir', type=str, default='../checkpoints', help='models are saved here')
	args = modify_args.run(parser)
	return args
