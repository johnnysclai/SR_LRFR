import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models import FNet, SRNet

import os
import numpy as np
from tqdm import tqdm
from common.loader import get_loader
from arguments import train_args

def freeze(model):
	for name, child in model.named_children():
		for param in child.parameters():
			param.requires_grad = False
		freeze(child)


def save_network(args, net, which_step):
	save_filename = 'edsr_lambda{}_step{}.pth'.format(args.lamb_id, which_step)
	save_dir = os.path.join(args.checkpoints_dir, args.name)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	save_path = os.path.join(save_dir, save_filename)
	if len(args.gpu_ids) > 1 and torch.cuda.is_available():
		try:
			torch.save(net.module.cpu().state_dict(), save_path)
		except:
			torch.save(net.cpu().state_dict(), save_path)
	else:
		torch.save(net.cpu().state_dict(), save_path)


def tensor2SFTensor(tensor):
	return tensor * (127.5 / 128.)


def main():
	args = train_args.get_args()
	dataloader = get_loader(args, 'celeba')
	train_iter = iter(dataloader)

	## Setup FNet
	fnet = getattr(FNet, 'sface')()
	fnet.load_state_dict(torch.load('../pretrained/sface.pth'))
	freeze(fnet)
	fnet.to(args.device)

	## Setup SRNet
	srnet = SRNet.edsr()
	srnet.to(args.device)
	if len(args.gpu_ids) > 1:
		srnet = nn.DataParallel(srnet)

	optimizer = optim.Adam(srnet.parameters(), lr=args.lr, betas=(0.9, 0.999))
	scheduler = StepLR(optimizer, step_size=args.decay_step, gamma=0.5)
	criterion_pixel = nn.L1Loss()

	losses = ['loss', 'loss_pixel', 'loss_feature', 'lr', 'index']
	pbar = tqdm(range(1, args.iterations + 1), ncols=0)
	for total_steps in pbar:
		srnet.train()
		scheduler.step()  # update learning rate
		lr = optimizer.param_groups[0]['lr']
		try:
			inputs = next(train_iter)
		except:
			train_iter = iter(dataloader)
			inputs = next(train_iter)
		index = np.random.randint(2, 4 + 1)
		lr_face = inputs['down{}'.format(2 ** index)].to(args.device)
		mr_face = inputs['down{}'.format(2 ** (index - 2))].to(args.device)
		if index == 2:
			hr_face = mr_face
		else:
			hr_face = inputs['down1'].to(args.device)
		sr_face = srnet(lr_face)
		loss_pixel = criterion_pixel(sr_face, mr_face.detach())
		loss = loss_pixel
		# Feature loss
		sr_face_up = nn.functional.interpolate(sr_face, size=(112, 96), mode='bilinear', align_corners=False)
		if args.lamb_id > 0:
			sr_face_feature = fnet(tensor2SFTensor(sr_face_up))
			hr_face_feature = fnet(tensor2SFTensor(hr_face)).detach()
			loss_feature = 1 - torch.nn.CosineSimilarity()(sr_face_feature, hr_face_feature)
			loss_feature = loss_feature.mean()
			loss += args.lamb_id * loss_feature
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# display
		description = ""
		for name in losses:
			try:
				value = float(eval(name))
				if name == 'index':
					description += '{}: {:.0f} '.format(name, value)
				elif name == 'lr':
					description += '{}: {:.3e} '.format(name, value)
				else:
					description += '{}: {:.3f} '.format(name, value)
			except:
				continue
		pbar.set_description(desc=description)

	# Save the final SR model
	save_network(args, srnet, args.iterations)


if __name__ == '__main__':
	main()
