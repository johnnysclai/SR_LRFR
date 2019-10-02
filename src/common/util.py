import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import numpy as np
import cv2
from .matlab_cp2tform import get_similarity_transform_for_cv2
import pandas as pd
import os
import sys
from scipy.spatial.distance import cdist


def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False


def KFold(n=6000, n_folds=10, shuffle=False):
	folds = []
	base = list(range(n))
	for i in range(n_folds):
		test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
		train = list(set(base) - set(test))
		folds.append([train, test])
	return folds


def eval_acc(threshold, diff):
	y_predict = np.int32(diff[:, 0] > threshold)
	y_true = np.int32(diff[:, 1])
	accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
	return accuracy


def find_best_threshold(thresholds, predicts):
	best_threshold = best_acc = 0
	for threshold in thresholds:
		accuracy = eval_acc(threshold, predicts)
		if accuracy >= best_acc:
			best_acc = accuracy
			best_threshold = threshold
	return best_threshold


def alignment(src_img, src_pts, size=None):
	ref_pts = [[30.2946, 51.6963], [65.5318, 51.5014],
	           [48.0252, 71.7366], [33.5493, 92.3655],
	           [62.7299, 92.2041]]
	if size is not None:
		ref_pts = np.array(ref_pts)
		ref_pts[:, 0] = ref_pts[:, 0] * size / 96
		ref_pts[:, 1] = ref_pts[:, 1] * size / 96
		crop_size = (int(size), int(112 / (96 / size)))
	else:
		crop_size = (96, 112)
	src_pts = np.array(src_pts).reshape(5, 2)
	s = np.array(src_pts).astype(np.float32)
	r = np.array(ref_pts).astype(np.float32)
	tfm = get_similarity_transform_for_cv2(s, r)
	face_img = cv2.warpAffine(src_img, tfm, crop_size, flags=cv2.INTER_CUBIC)
	if size is not None:
		face_img = cv2.resize(face_img, dsize=(96, 112), interpolation=cv2.INTER_CUBIC)
	return face_img


class L2Norm(nn.Module):
	def forward(self, input, dim=1):
		return F.normalize(input, p=2, dim=dim)


def face_ToTensor(img):
	return (ToTensor()(img) - 0.5) * 2


def tensor_sface_norm(tensor):
	# tensor in range [-1, 1] -> [-0.99609375, 0.99609375]
	return tensor * (127.5 / 128.)


def tensor_pair_cosine_distance(features11, features12, features21, features22, type='normal'):
	if type == 'concat':
		features1 = torch.cat((features11, features12), dim=1)
		features2 = torch.cat((features21, features22), dim=1)
	elif type == 'sum':
		features1 = features11 + features12
		features2 = features21 + features22
	elif type == 'normal':
		features1 = features11
		features2 = features21
	else:
		print('tensor_pair_cosine_distance unspported type!')
		sys.exit()
	scores = torch.nn.CosineSimilarity()(features1, features2)
	scores = scores.cpu().numpy().reshape(-1, 1)
	return scores


def tensors_cvBicubic_resize(args, tensors):
	_, _, h_, w_ = tensors.shape
	if h_ == args.h and w_ == args.w:
		return tensors
	else:
		image = np.uint8((tensors.cpu().numpy() + 1) * 0.5 * 255.0)
		image = np.transpose(image, (0, 2, 3, 1))
		image = [cv2.resize(img, dsize=(args.w, args.h), interpolation=cv2.INTER_CUBIC) for img in image]
		image = np.asarray(image)
		image = np.float32(np.transpose(image, (0, 3, 1, 2)))
		image = torch.Tensor(image).to(args.device)
		image = (image - 127.5) / 127.5
		return image
