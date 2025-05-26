import cv2
import numpy as np
import scipy.signal as sg


patterns = {
	'GP': 'bggr',
	'IP': 'rggb',
	'S6': 'grbg',
	'N6': 'bggr',
	'G4': 'bggr'
}
def get_pattern(img_name):
	if 'S6' in img_name:
		return patterns['S6']
	elif 'GP' in img_name:
		return patterns['GP']
	elif 'N6' in img_name:
		return patterns['N6']
	elif 'IP' in img_name:
		return patterns['IP']
	elif 'G4' in img_name:
		return patterns['G4']
	else:
		return None

def get_pattern_list(img_name):
	if 'S6' in img_name:
		return [1, 0, 2, 1]
	elif 'GP' in img_name:
		return [2, 1, 1, 0]
	elif 'N6' in img_name:
		return [2, 1, 1, 0]
	elif 'IP' in img_name:
		return [0, 1, 1, 2]
	elif 'G4' in img_name:
		return [2, 1, 1, 0]
	else:
		return None

def white_balance(normalized_image, as_shot_neutral, cfa_pattern):
	# if type(as_shot_neutral[0]) is Ratio:
	# 	as_shot_neutral = ratios2floats(as_shot_neutral)
	idx2by2 = [[0, 0], [0, 1], [1, 0], [1, 1]]
	step2 = 2
	white_balanced_image = np.zeros(normalized_image.shape)
	for i, idx in enumerate(idx2by2):
		idx_y = idx[0]
		idx_x = idx[1]
		white_balanced_image[idx_y::step2, idx_x::step2] = normalized_image[idx_y::step2, idx_x::step2] / as_shot_neutral[cfa_pattern[i]]
	white_balanced_image = np.clip(white_balanced_image, 0.0, 1.0)
	return white_balanced_image

def get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type='VNG'):
	# using opencv edge-aware demosaicing
	if alg_type != '':
		alg_type = '_' + alg_type
	if output_channel_order == 'BGR':
		if cfa_pattern == [0, 1, 1, 2]:  # RGGB
			opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
		elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
			opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2BGR' + alg_type)
		elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
			opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2BGR' + alg_type)
		elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
			opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2BGR' + alg_type)
		else:
			opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2BGR' + alg_type)
			print("CFA pattern not identified.")
	else:  # RGB
		if cfa_pattern == [0, 1, 1, 2]:  # RGGB
				opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
		elif cfa_pattern == [2, 1, 1, 0]:  # BGGR
			opencv_demosaic_flag = eval('cv2.COLOR_BAYER_RG2RGB' + alg_type)
		elif cfa_pattern == [1, 0, 2, 1]:  # GRBG
			opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GB2RGB' + alg_type)
		elif cfa_pattern == [1, 2, 0, 1]:  # GBRG
			opencv_demosaic_flag = eval('cv2.COLOR_BAYER_GR2RGB' + alg_type)
		else:
			opencv_demosaic_flag = eval('cv2.COLOR_BAYER_BG2RGB' + alg_type)
			print("CFA pattern not identified.")
	return opencv_demosaic_flag

def demosaic(white_balanced_image, cfa_pattern, output_channel_order='BGR', alg_type='VNG'):
	"""
	Demosaic a Bayer image.
	:param white_balanced_image:
	:param cfa_pattern:
	:param output_channel_order:
	:param alg_type: algorithm type. options: '', 'EA' for edge-aware, 'VNG' for variable number of gradients
	:return: Demosaiced image
	"""
	if alg_type == 'VNG':
		max_val = 255
		wb_image = (white_balanced_image * max_val).astype(dtype=np.uint8)
	else:
		max_val = 16383
		wb_image = (white_balanced_image * max_val).astype(dtype=np.uint16)

	if alg_type in ['', 'EA', 'VNG']:
		opencv_demosaic_flag = get_opencv_demsaic_flag(cfa_pattern, output_channel_order, alg_type=alg_type)
		demosaiced_image = cv2.cvtColor(wb_image, opencv_demosaic_flag)
	elif alg_type == 'menon2007':
		raise NotImplementedError()
		pass

	demosaiced_image = demosaiced_image.astype(dtype=np.float32) / max_val

	return demosaiced_image

def apply_color_space_transform(rgb_image, color_matrix_1=None, color_matrix_2=None):
	color_matrix = color_matrix_2.reshape(3, 3)
	color_matrix = color_matrix / np.sum(color_matrix, axis=1, keepdims=True)
	color_matrix = np.linalg.inv(color_matrix)

	xyz_image = rgb_image @ (color_matrix.T)
	xyz_image = np.clip(xyz_image, 0, 1)
	
	return xyz_image

def transform_xyz_to_srgb(xyz_image):
	M = np.array([
		[3.2404542, -1.5371385, -0.4985314],
		[-0.9692660, 1.8760108, 0.0415560],
		[0.0556434, -0.2040259, 1.0572252]
	])

	rgb_linear = xyz_image @ (M.T)
	rgb_linear = np.clip(rgb_linear, 0, 1)

	mask_lin = rgb_linear < 0.0031308
	rgb_gamma = 1.055 * np.power(rgb_linear, 1/2.4) - 0.055
	rgb_lin = 12.92 * rgb_linear
	srgb_image = rgb_gamma * (1-mask_lin) + rgb_lin * mask_lin
	srgb_image = np.clip(srgb_image, 0, 1)
	return srgb_image

def fix_orientation(image, orientation):
	# 1 = Horizontal(normal)
	# 2 = Mirror horizontal
	# 3 = Rotate 180
	# 4 = Mirror vertical
	# 5 = Mirror horizontal and rotate 270 CW
	# 6 = Rotate 90 CW
	# 7 = Mirror horizontal and rotate 90 CW
	# 8 = Rotate 270 CW

	if type(orientation) is list:
		orientation = orientation[0]

	if orientation == 1:
		pass
	elif orientation == 2:
		image = cv2.flip(image, 0)
	elif orientation == 3:
		image = cv2.rotate(image, cv2.ROTATE_180)
	elif orientation == 4:
		image = cv2.flip(image, 1)
	elif orientation == 5:
		image = cv2.flip(image, 0)
		image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
	elif orientation == 6:
		image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	elif orientation == 7:
		image = cv2.flip(image, 0)
		image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	elif orientation == 8:
		image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

	return image

def apply_gamma(x):
	return x ** (1.0 / 2.2)

def apply_tone_map(x):
	# simple tone curve
	return 3 * x ** 2 - 2 * x ** 3

	# tone_curve = loadmat('tone_curve.mat')
	# tone_curve = loadmat(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tone_curve.mat'))
	# tone_curve = sio.loadmat("C:\\Users\\swHong\\work\\sidd_pipeline\\simple-camera-pipeline\\python\\tone_curve.mat")
	# tone_curve = tone_curve['tc']
	# x = np.round(x * (len(tone_curve) - 1)).astype(int)
	# tone_mapped_image = np.squeeze(tone_curve[x])
	# return tone_mapped_image

def get_bayer_mask(shape, pattern='rggb'):
  r_idx = pattern.index('r')
  b_idx = pattern.index('b')
  gr_idx = r_idx-1 if r_idx%2 else r_idx+1
  gb_idx = b_idx-1 if b_idx%2 else b_idx+1

  _pattern = np.array(['__' for _ in range(4)])
  _pattern[[r_idx, gr_idx, gb_idx, b_idx]] = ['r', 'gr', 'gb', 'b']
  pattern = list(_pattern)

  channels = dict((channel, np.zeros(shape)) for channel in ['r', 'gr', 'gb', 'b'])
  for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
    channels[channel][y::2, x::2] = 1
		
  return tuple(channels[c].astype(np.int32) for c in ['r', 'gr', 'gb', 'b'])

def demosaic_matlab(image, pattern):
	mask_r, mask_gr, mask_gb, mask_b = get_bayer_mask(image.shape, pattern)
	h_g = np.array([
		[ 0,  0, -1,  0,  0],
		[ 0,  0,  2,  0,  0],
		[-1,  2,  4,  2, -1],
		[ 0,  0,  2,  0,  0],
		[ 0,  0, -1,  0,  0]])
	h_rb_h = np.array([
		[  0,  0, 0.5,  0,   0],
		[  0, -1,   0, -1,   0],
		[ -1,  4,   5,  4,  -1],
		[  0, -1,   0, -1,   0],
		[  0,  0, 0.5,  0,   0]])
	h_rb_v = h_rb_h.T
	h_rb_omni = np.array([
		[0, 0, -1.5, 0, 0],
		[0, 2, 0, 2, 0],
		[-1.5, 0, 6, 0, -1.5],
		[0, 2, 0, 2, 0],
		[0, 0, -1.5, 0, 0]])
	img_pad = np.pad(image, 2, 'reflect').astype(np.int32)
	
	# green interpolation
	g_interp = np.right_shift(sg.convolve2d(img_pad, h_g, 'valid'), 3)
	g = image * (mask_gr+mask_gb) + np.clip(g_interp, 0, (2**16)) * (mask_r+mask_b)

	# red, blue interpolation
	rb_interp_h = np.right_shift(sg.convolve2d(img_pad, (h_rb_h*2).astype(np.int32), 'valid'), 4)
	rb_interp_v = np.right_shift(sg.convolve2d(img_pad, (h_rb_v*2).astype(np.int32), 'valid'), 4)
	rb_interp_hv = np.right_shift(sg.convolve2d(img_pad, (h_rb_omni*2).astype(np.int32), 'valid'), 4)
	# rb_interp_h = np.floor(sg.convolve2d(img_pad, (h_rb_h*2).astype(np.int32), 'valid')/16).astype(np.int32)
	# rb_interp_v = np.floor(sg.convolve2d(img_pad, (h_rb_v*2).astype(np.int32), 'valid')/16).astype(np.int32)
	# rb_interp_hv = np.floor(sg.convolve2d(img_pad, (h_rb_omni*2).astype(np.int32), 'valid')/16).astype(np.int32)

	rb_interp_h = np.clip(rb_interp_h, 0, (2**16)).astype(np.uint32)
	rb_interp_v = np.clip(rb_interp_v, 0, (2**16)).astype(np.uint32)
	rb_interp_hv = np.clip(rb_interp_hv, 0, (2**16)).astype(np.uint32)

	r = mask_gr * rb_interp_h + mask_gb * rb_interp_v + mask_b * rb_interp_hv + mask_r * image
	b = mask_gb * rb_interp_h + mask_gr * rb_interp_v + mask_r * rb_interp_hv + mask_b * image

	return np.dstack([r, g, b])

def gt_pipeline(normalized_image, metadata, img_name):
	cfa_pattern = get_pattern_list(img_name)
	pattern = get_pattern(img_name)
	whitelevel = metadata['AsShotNeutral'][0]
	color_matrix = metadata['ColorMatrix2'].astype(np.float64)

	wb_out = white_balance(normalized_image, whitelevel, cfa_pattern)

	temp = np.clip(wb_out, 0, 1)
	temp = np.round(np.clip(wb_out*(2**16), 0, (2**16)-1)).astype(np.uint16)
	demosaic_int = demosaic_matlab(temp, pattern)
	demosaic_out = demosaic_int.astype(np.float32)/(2**16)

	xyz_image = apply_color_space_transform(demosaic_out, color_matrix_2=color_matrix)
	srgb_out = transform_xyz_to_srgb(xyz_image)
	return srgb_out