import argparse
from common.util import str2bool


def get_args():
	parser = argparse.ArgumentParser(description='SR LRFR')
	parser.add_argument('--fnet', default='sface', type=str, help='sface')
	parser.add_argument('--fnet_pth', default='../pretrained/sface.pth', type=str, help='')
	parser.add_argument('--srnet', default='edsr', type=str, help='')
	parser.add_argument('--lfw_bs', default=128, type=int, help='LFW batch size')
	parser.add_argument('--gpu_ids', default='0', type=str, help='gpu ids: e.g. 0 or 0,1 or 0,2')
	return parser
