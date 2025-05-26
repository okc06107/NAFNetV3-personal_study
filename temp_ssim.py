import numpy as np
import cv2
import torch

def _generate_3d_gaussian_kernel():
  kernel = cv2.getGaussianKernel(11, 1.5)
  window = np.outer(kernel, kernel.transpose())
  kernel_3 = cv2.getGaussianKernel(11, 1.5)
  kernel = torch.tensor(np.stack([window * k for k in kernel_3], axis=0))
  conv3d = torch.nn.Conv3d(1, 1, (11, 11, 11), stride=1, padding=(5, 5, 5), bias=False, padding_mode='replicate')
  conv3d.weight.requires_grad = False
  conv3d.weight[0, 0, :, :, :] = kernel
  return conv3d

def _3d_gaussian_calculator(img, conv3d):
  out = conv3d(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
  return out

def calculate_ssim(img1:np.ndarray, img2:np.ndarray, crop_border:int=0, max_value=255):
  assert(img1.shape == img2.shape)
  if crop_border != 0:
    img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
    img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
  
  ssims = []
  assert(img1.max() <= max_value and img2.max() <= max_value)

  C1 = (0.01 * max_value) ** 2
  C2 = (0.03 * max_value) ** 2
  
  kernel = _generate_3d_gaussian_kernel()
  
  mu1 = _3d_gaussian_calculator(img1, kernel)
  mu2 = _3d_gaussian_calculator(img2, kernel)
  mu1_sq = mu1 ** 2
  mu2_sq = mu2 ** 2
  mu1_mu2 = mu1 * mu2
  sigma1_sq = _3d_gaussian_calculator(img1**2, kernel) - mu1_sq
  sigma2_sq = _3d_gaussian_calculator(img2**2, kernel) - mu2_sq
  sigma12 = _3d_gaussian_calculator(img1 * img2, kernel) - mu1_mu2
  ssim_map = ((2*mu1_mu2+C1) * (2*sigma12 + C2)) / ((mu1_sq+mu2_sq+C1) * (sigma1_sq+sigma2_sq+C2))
  return float(ssim_map.mean())