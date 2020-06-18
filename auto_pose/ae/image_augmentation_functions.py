


from math import pi

#import matplotlib
#<matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import tensorflow as tf 
import numpy as np 

def map_decorator(name):
	def real_decorator(fn):
		def fn_wrapper(*argv, **kwargs):
			
			new_argv = ()
			new_kwargs = {}
			i=0
			if argv is not ():
				for arg in argv:
					if i == 0:
						values = arg
					else:
						new_argv += (arg,)
					i += 1
			for key, item in list(kwargs.items()):
				#print(key)
				if key == 'in_tensor':
					values = item
				else:
					new_kwargs[key] = item

			tensor_shape = values.get_shape().as_list()
			with tf.variable_scope(name):
				if len(tensor_shape) == 3:
					values = tf.expand_dims(values, 0)
				result = fn(values, *new_argv, **new_kwargs)
				if len(tensor_shape) == 3:
					result = tf.squeeze(result, axis=0)
				return result
		
		return fn_wrapper
	return real_decorator

@map_decorator('reshape_and_pad')
def reshape_and_pad(in_tensor, zoom_value, mode):
	""" zoom out of the image by first reshaping and padding after

	zooms out of an image to reduce the Area of the image by a factor defined by zoom_value

	Args:
		in_tensor:	4D Tensor with shape (batch_size, height, width, channels), which gets zoomed out of.
		zoom_value:	1D Tensor which defines the factor of the size of the zoomed image. 
		mode:		Mode for Padding. 'CONSTANT' for zeropadding, 'REFLECT' for reflecting the image
					elements without the outermost pixel. 'SYMMETRIC' for reflecting the image elements
					with the outermost pixel.

	Returns:
		The zoomed image with the same size as the input image. 

	"""
	#calculate the new image shape to which to resize before padding the image again 
	in_tensor_shape = in_tensor.get_shape()
	in_tensor_shape_list = in_tensor_shape.as_list()

	new_size1 = tf.cast(tf.multiply(tf.cast(in_tensor_shape[1], tf.float32), tf.sqrt(zoom_value)), tf.int32)
	new_size2 = tf.cast(tf.multiply(tf.cast(in_tensor_shape[2], tf.float32), tf.sqrt(zoom_value)), tf.int32)
	new_size = [new_size1, new_size2]

	#resize the image
	img = tf.image.resize_images(in_tensor, new_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		
	#calculate the padding values along each dimension and create the list of all paddings
	#channels and batch size remain unchanged
	pad1 = tf.subtract(in_tensor_shape[1], new_size[0])
	pad2 = tf.subtract(in_tensor_shape[2], new_size[1])
	padding = [[0, 0],
		[tf.cast(tf.floordiv(pad1, 2), dtype=tf.int32) + tf.floormod(pad1, 2), tf.cast(tf.floordiv(pad1, 2), dtype=tf.int32)],
		[tf.cast(tf.floordiv(pad2, 2), dtype=tf.int32) + tf.floormod(pad2, 2), tf.cast(tf.floordiv(pad2, 2), dtype=tf.int32)], 
		[0, 0]]
	#pad the image
	img = tf.pad(img, paddings = padding, mode = mode)
	#set the shape of the return tensor manually back to the original shape
	img.set_shape(in_tensor_shape)
	#return the result image

	return img

@map_decorator('crop_and_reshape')
def crop_and_reshape(in_tensor, zoom_value):
	""" crop the image and resize afterwards for in zooming

	crops the image and then resizes with nearest neighbor back to the original image. Performs basically
	an in zooming operation

	Args:
		in_tensor:	4D Tensor of shape (batch_size, height, width, channels). operation gets performed equally
					on the whole batch
		zoom_value:	1D Tensor which defines the factor of the new image area size. Must be greater or equal than 1.0

	Returns:
		The zoomed in tensor with the same shape as the input tensor.

	"""
	
	in_tensor_shape = in_tensor.get_shape()
	in_tensor_shape_list = in_tensor_shape.as_list()

	#Calculate the shape to which to crop
	crop_factor = tf.truediv(1.0, tf.sqrt(zoom_value))
	new_size1 = tf.cast(tf.multiply(tf.cast(in_tensor_shape[1], tf.float32), crop_factor), tf.int32)
	new_size2 = tf.cast(tf.multiply(tf.cast(in_tensor_shape[2], tf.float32), crop_factor), tf.int32)

	#Calculate the offset values of the cropping bounding box. Always crop the central part
	crop_offset_height = tf.cast(tf.floordiv(tf.subtract(in_tensor_shape[1], new_size1), 2), tf.int32)
	crop_offset_width = tf.cast(tf.floordiv(tf.subtract(in_tensor_shape[2], new_size2), 2), tf.int32)

	#crop the image to the bounding box, previously defined
	img = tf.image.crop_to_bounding_box(in_tensor,
					crop_offset_height,
					crop_offset_width,
					new_size1,
					new_size2)

	#resize with nearest neighbor interpolation
	img = tf.image.resize_images(img, [in_tensor_shape[1], in_tensor_shape[2]], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	#set the shape of the new tensor manually to the original shape
	img.set_shape(in_tensor_shape)
	#return the zoomed tensor.
	return img

@map_decorator('zoom_image_object')
def zoom_image_object(in_tensor, zoom_range, padding_mode = 'CONSTANT'):
	""" Zoom image objects 
	Perform zentral zoom on an image. Image size is preserved. For a zoomfactor
	less than one the image gets padded with zeros. Else it is cropped

	Args:
		in_tensor:	4D Tensor with the format (batch_size, image_height, image_width, channels)
					zoom gets equally performed on all elements of the batch

		zoom_value:	1D Tensor, contains the size factor of the Area of the new image, to which 
					the object shall be zoomed. 
		padding_mode:	The padding mode, if zoom_value is less than 1. 

	Returns:
		The zoomed image with the same shape as the input tensor. Image gets automatically cropped
		or padded depending on the factor. 
	"""
	
	zoom_value = tf.random_shuffle(zoom_range)[0]
	img = tf.case([(tf.greater(zoom_value, 1.0), lambda : crop_and_reshape(in_tensor, zoom_value))], 
					default=lambda : reshape_and_pad(in_tensor, zoom_value, padding_mode))
	return img

@map_decorator('translate_im_obj')
def translate_im_obj(in_tensor, trans_range):
	translation = tf.to_float(tf.random_uniform([in_tensor.get_shape().as_list()[0], 2], 
						trans_range[0],
						trans_range[1],
						dtype = tf.int32))
	return tf.contrib.image.translate(in_tensor, translations = translation, interpolation = 'NEAREST')

def sign(x):
	if x < 0:
		return -1
	else:
		return 1

@map_decorator('zoom_and_translate')
def zoom_and_translate(in_tensor, zoom_range, trans_range):
	with tf.variable_scope('zoom_image_object'):
		zoom_value = tf.random_shuffle(zoom_range)[0]
		img = tf.case([(tf.greater(zoom_value, 1.0), lambda : crop_and_reshape(in_tensor, zoom_value))], 
						default=lambda : reshape_and_pad(in_tensor, zoom_value, 'CONSTANT'))
		shape = in_tensor.get_shape().as_list()[1:3]
	new_trans_range = [tf.to_int32(sign(x) * (tf.abs(1.0 - tf.sqrt(zoom_value))) * shape[i] + x) for i, x in enumerate(trans_range)]
	
	img = translate_im_obj(img, new_trans_range)
	img.set_shape(in_tensor.get_shape())
	return img

@map_decorator('in_plane_rotation')
def in_plane_rotation(in_tensor):
	rad_rot = tf.random_uniform([], 0, 2 * pi)
	return tf.contrib.image.rotate(in_tensor, rad_rot)

@map_decorator('add_black_patches')
def add_black_patches(in_tensor, max_area_cov = 0.25, max_patch_nb = 5):
	
	@map_decorator('square_patch')
	def square_patch(in_tensor, max_coverage = 0.05):
		
		in_tensor_shape = in_tensor.get_shape()
		in_tensor_shape_list = in_tensor.get_shape().as_list()
		print(in_tensor_shape)
		print(in_tensor_shape_list)
		
		coverage = tf.random_uniform([], minval = 0.01, maxval = max_coverage)

		patch_edge_len = tf.cast(tf.multiply(tf.sqrt(coverage), 
				tf.minimum(tf.to_float(in_tensor_shape[1]), tf.to_float(in_tensor_shape[2]))), dtype=tf.int32)
			
		max_x = tf.subtract(in_tensor_shape[1], patch_edge_len)
		max_y = tf.subtract(in_tensor_shape[2], patch_edge_len)


		# shape = in_tensor.get_shape().as_list()
		# #print(shape)
		# brightness_offset = tf.random_uniform([shape[0]], minval = -max_offset, maxval = max_offset, dtype = tf.float32)
		
		x_offset = tf.random_uniform([in_tensor_shape_list[0], 1], 0, max_x, dtype = tf.int32)
		y_offset = tf.random_uniform([in_tensor_shape_list[0], 1], 0, max_y, dtype = tf.int32)

		patch = tf.ones([in_tensor_shape[0], patch_edge_len, patch_edge_len, in_tensor_shape[3]], dtype=tf.float32)
		#Pad to start position, then translate
		mask = tf.image.pad_to_bounding_box(patch, 0, 0, in_tensor_shape_list[1], in_tensor_shape_list[2])

		mask = tf.contrib.image.translate(mask, 
					translations = tf.to_float(tf.concat([x_offset, y_offset], axis = 1)), 
					interpolation = 'NEAREST')

		mask = in_plane_rotation(mask)

		return tf.multiply(in_tensor, 1.0 - mask)

	@map_decorator('circle_patch')
	def circle_patch(in_tensor, max_coverage = 0.05):
		"""
		Create a circlic occlusion patch in the input tensor on a random position
		
		Input:
			in_tensor:	A Tensor input, on which the patch should be applied. Must either be 3-dimensional with the 
					dimensions (Height, Width, Channels) or 4-dimensional with the dimensions (Batch Size, Height, Width, Channels)
			coverage:: 	percentage of how much of the image should be covered by the patch. Must be between 0 and 1

		Output:
			returns the original input image with a random circlic black occlusion patch on it.
		"""
		in_tensor_shape = in_tensor.get_shape()
		in_tensor_shape_list = in_tensor.get_shape().as_list()
		coverage = tf.random_uniform([], minval = 0.01, maxval = max_coverage)

		# Calculate the diameter of the circle out of the coverage
		# This is done by calculating the squareroot of the area coverage and multiplying it with the smaller 
		# dimension (either Height or Width) of the input tensor
		patch_edge_len = tf.cast(tf.multiply(tf.sqrt(coverage), 
				tf.minimum(tf.to_float(in_tensor_shape[1]), tf.to_float(in_tensor_shape[2]))), dtype=tf.int32)
		# patch_edge_len = 20

		# Search for the pixel out of the boundaries and turn them black
		# Calculate the radius. Starting point is in the mid of a pixel, end point is the half of the diameter
		radius = tf.to_float(patch_edge_len) / 2.0 - 0.5

		# Create x and y pixel position masks ranging from 0 to the diameter of the circle
		x_pos = tf.ones([patch_edge_len, patch_edge_len]) * tf.range(0.0, tf.to_float(patch_edge_len), 1.0, dtype=tf.float32)
		y_pos = tf.transpose(x_pos)

		# Calculate the distance of every pixel to the centerpoint of the circle and expand the dimensions to the input tensor dimension
		distance_to_mid = tf.sqrt(tf.add(tf.square(tf.subtract(x_pos, radius)), tf.square(tf.subtract(y_pos, radius))))
		distance_to_mid = tf.expand_dims(distance_to_mid, axis=0)
		distance_to_mid = tf.tile(tf.expand_dims(distance_to_mid, axis = 3), [in_tensor_shape_list[0], 1, 1, in_tensor_shape_list[3]])

		# Create the circlic patch --> 0, if the distance to the center is greater than the radius, 1 otherwise
		patch = tf.where(tf.greater(distance_to_mid, radius),
				tf.zeros([in_tensor_shape[0], patch_edge_len, patch_edge_len, in_tensor_shape[3]], dtype=tf.float32), 
				tf.ones([in_tensor_shape[0], patch_edge_len, patch_edge_len, in_tensor_shape[3]], dtype = tf.float32))

		# Create the image mask out of the patch --> has the same size as the input tensor
		mask = tf.image.pad_to_bounding_box(patch, 0, 0, in_tensor_shape_list[1], in_tensor_shape_list[2])

		# Move the mask element in the image to a random position
		max_x = tf.subtract(in_tensor_shape_list[1], patch_edge_len)
		max_y = tf.subtract(in_tensor_shape_list[2], patch_edge_len)
		x_offset = tf.random_uniform([in_tensor_shape_list[0], 1], 0, max_x, dtype = tf.int32)
		y_offset = tf.random_uniform([in_tensor_shape_list[0], 1], 0, max_y, dtype = tf.int32)

		mask = tf.contrib.image.translate(mask,
				translations = tf.to_float(tf.concat([x_offset, y_offset], axis = 1)),
				interpolation = 'NEAREST')

		# return the input multiplied by the inverted mask
		return tf.multiply(in_tensor, 1.0 - mask)

	def no_patch(in_tensor):
		return in_tensor

	modified_tensor = in_tensor
	choices = 3
	patches = 5
	max_coverage = max_area_cov / patches
	patch_1 = lambda: square_patch(modified_tensor, max_coverage)
	patch_2 = lambda: circle_patch(modified_tensor, max_coverage)
	patch_default = lambda: no_patch(modified_tensor)

	for i in range(patches):
		prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)

		modified_tensor = tf.case([(tf.less(prob, 1.0/choices), patch_1), 
				(tf.logical_and(tf.greater_equal(prob, 1.0/choices), tf.less(prob, 2.0/choices)), patch_2)],
				default = patch_default)
		modified_tensor.set_shape(in_tensor.get_shape())

	return modified_tensor	

@map_decorator('gaussian_noise')
def gaussian_noise(in_tensor, stddev = 0.01):
	prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)
	noise  = tf.case([(tf.less(prob, 0.5), lambda: tf.random_normal(in_tensor.get_shape().as_list(), mean = 0.0, stddev = stddev, dtype = tf.float32))],
		default = lambda: tf.constant(0, shape = in_tensor.get_shape(), dtype = tf.float32))
	return tf.maximum(tf.minimum(in_tensor+noise, 1.0), 0.0)

def add_background(obj_images, background_images, bg_color = 0, name='add_background'):
	with tf.variable_scope(name):
		if bg_color == 0:
			mask = tf.where(tf.greater_equal(obj_images, 0.05), 
				tf.ones_like(obj_images), 
				tf.zeros_like(obj_images))
			return tf.where(tf.equal(mask, 0.0), background_images,	obj_images)
		else:
			mask = tf.where(tf.less_equal(obj_images, 0.95),
				tf.ones_like(obj_images),
				tf.zeros_like(obj_images))
			return tf.where(tf.equal(mask, 0.0), background_images, obj_images)

@map_decorator('invert_color')
def invert_color(in_tensor):

	def _all_channels(in_tensor):
		return 1.0 - in_tensor

	def _one_channel(in_tensor):
		channel = tf.random_uniform([], minval = 0, maxval = 3, dtype = tf.int32)
		c = in_tensor.get_shape().as_list()[3]
		channels = tf.unstack(in_tensor, c, 3)
		new_channels = []
		for i, tensor_channel in enumerate(channels):
			new_channels.append(tf.where(tf.equal(i, channel),
				1.0 - tensor_channel,
				tensor_channel))

		return tf.stack(new_channels, 3)

	def _no_aug(in_tensor):
		return in_tensor

	prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)

	aug1 = lambda: _all_channels(in_tensor)
	aug2 = lambda: _one_channel(in_tensor)
	aug3 = lambda: _no_aug(in_tensor)

	# return tf.case([(tf.less_equal(prob, 0.1), aug1), 
	# 	(tf.less_equal(prob, 0.3), aug2)],
	# 	default = aug3)
	return tf.case([(tf.less_equal(prob, 0.25), aug1), 
		(tf.less_equal(prob, 0.5), aug2)],
		default = aug3)


@map_decorator('invert_color_all')
def invert_color_all(in_tensor):

	def _all_channels(in_tensor):
		return 1.0 - in_tensor

	def _no_aug(in_tensor):
		return in_tensor

	prob = tf.random_uniform([], minval=0.0, maxval=1.0, dtype=tf.float32)

	def aug1(): return _all_channels(in_tensor)
	def aug3(): return _no_aug(in_tensor)

	# return tf.case([(tf.less_equal(prob, 0.1), aug1),
	# 	(tf.less_equal(prob, 0.3), aug2)],
	# 	default = aug3)
	return tf.case([(tf.less_equal(prob, 0.25), aug1),
                 (tf.less_equal(prob, 0.4), aug1)],
                default=aug3)
	
@map_decorator('random_brightness')
def random_brightness(in_tensor, max_offset):
	
	def aug_all_color_channels(in_tensor, max_offset):
		shape = in_tensor.get_shape().as_list()
		#print(shape)
		brightness_offset = tf.random_uniform([shape[0]], minval = -max_offset, maxval = max_offset, dtype = tf.float32)
		brightness_offset = tf.tile(brightness_offset[:, tf.newaxis, tf.newaxis, tf.newaxis], [1, shape[1], shape[2], shape[3]])
		return in_tensor+brightness_offset

	def aug_one_color_channel(in_tensor, max_offset):
		shape = in_tensor.get_shape().as_list()
		#print(shape)
		brightness_offset = tf.random_uniform([shape[0]], minval = -max_offset, maxval = max_offset, dtype = tf.float32)
		
		offset_array = tf.tile(brightness_offset[:, tf.newaxis, tf.newaxis, tf.newaxis], [1, shape[1], shape[2], 1])
		zero_array = tf.constant(0.0, shape = shape[:3] + [1], dtype = tf.float32)

		channel = tf.random_uniform([], minval = 0, maxval = 3, dtype = tf.int32)
		offset = tf.case([(tf.equal(channel, 0), lambda: tf.concat([offset_array, zero_array, zero_array], 3)),
			(tf.equal(channel, 1), lambda: tf.concat([zero_array, offset_array, zero_array], 3))],
			default = lambda: tf.concat([zero_array, zero_array, offset_array], 3))
		#print offset.get_shape().as_list()
		return in_tensor + offset	

	def aug_no_color(in_tensor):
		return in_tensor

	shape = in_tensor.get_shape()
	prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)
	
	color1 = lambda: aug_all_color_channels(in_tensor, max_offset)
	color2 = lambda: aug_one_color_channel(in_tensor, max_offset)
	color3 = lambda: aug_no_color(in_tensor)

	#aug_tensor = tf.case([(tf.less_equal(prob, 0.33), color1),
	#	(tf.less_equal(prob, 0.66), color2)],
	#	default = color3)
	# aug_tensor = tf.case([(tf.less_equal(prob, 0.3), color1),(tf.less_equal(prob, 0.5), color2)],
	# 	default = color3)

	# prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)
	aug_tensor = tf.case([(tf.less_equal(prob, 0.5), color1)],default = color3)

	prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)
	aug_tensor = tf.case([(tf.less_equal(prob, 0.5), color2)],default = color3)

	aug_tensor.set_shape(shape)
	return tf.maximum(tf.minimum(aug_tensor, 1.0), 0.0)

@map_decorator('add_background_and_brightness_augment')
def add_background_and_brightness_augment(in_tensor, bg_images, max_offset, bg_color=0):
	if bg_color==0:
		mask = tf.where(tf.greater_equal(in_tensor, 0.05), 
			tf.ones_like(in_tensor), 
			tf.zeros_like(in_tensor))
	else:
		mask = tf.where(tf.less_equal(in_tensor, 0.95),
			tf.ones_like(in_tensor),
			tf.zeros_like(in_tensor))

	img_with_bg = random_brightness(tf.where(tf.equal(mask, 0.0), 
		random_brightness(bg_images, max_offset), 
		random_brightness(in_tensor, max_offset)), max_offset)

	#img_with_bg = random_brightness(tf.where(tf.equal(mask, 0.0), 
	#	bg_images, 
	#	in_tensor), max_offset)

	return img_with_bg


@map_decorator('gaussian_blur')
def gaussian_blur(in_tensor, size = 5):
	"""Makes 2D gaussian Kernel for convolution."""
	def _blur(in_tensor, size=5):
		std = tf.random_uniform([], 0.0, 1.0, dtype = tf.float32)
		
		d = tf.distributions.Normal(0.0, std)

		vals = d.prob(tf.range(start = tf.ceil(-size/2), limit = tf.floor(size/2) + 1, dtype = tf.float32))

		gauss_kernel = tf.einsum('i,j->ij', vals, vals)

		normalized_gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)

		normalized_gauss_kernel = normalized_gauss_kernel[:, :, tf.newaxis, tf.newaxis]

		# normalized_gauss_kernel = tf.stack([normalized_gauss_kernel,normalized_gauss_kernel,normalized_gauss_kernel],axis=2)
		tensor_channels = tf.unstack(in_tensor, num = 3, axis = 3)
		new_channels = []
		for c in tensor_channels:
			print (c.shape)
			out_channel = tf.nn.conv2d(c[:, :, :, tf.newaxis], normalized_gauss_kernel, strides = [1, 1, 1, 1], use_cudnn_on_gpu=False, data_format='NHWC',
 padding = 'SAME')
			print(out_channel.shape)
			new_channels.append(out_channel)
		out = tf.concat(new_channels, axis=3)
		print(out.shape)
		return out
		# in_tensor = tf.nn.conv2d(in_tensor, normalized_gauss_kernel, strides = [1,1,1,1], padding="SAME")
		# return in_tensor
	def _no_blur(in_tensor, size = 5):
		return in_tensor


	prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)

	return tf.case([(tf.less_equal(prob, 0.5), lambda: _blur(in_tensor, size))],
		default = lambda: _no_blur(in_tensor, size))

@map_decorator('contrast_normalization')
def contrast_normalization(in_tensor, f = [0.5, 2.0]):
	'''
	Perform Contrast Normalization on the input tensor
	The algorithm takes the mean value of the image
	and a factor and then factorizes the distance of each 
	pixel to the mean value to said factor

	example: if f=0.5, and mean is 128, then 256 becomes 192,
	0 becomes 64, etc.

	Lower f means lower contrast, higher f means higher contrast

	Inputs:
		in_tensor:	tensor to be normalized. Must
				either be 3D with format HxWxC or
				4D with format NxHxWxC (N - Batch size
				H - Height, W - Width, C - Channels)
		f:	normalization factor. normal range lies 
				between 0.5 and 2.0
	Returns:
		The normalized image batch
	'''
	def _normalize(in_tensor, f):

		shape = in_tensor.get_shape().as_list()
		factor = tf.tile(tf.random_uniform([shape[0]], f[0], f[1], dtype = tf.float32)[:, tf.newaxis, tf.newaxis, tf.newaxis],
			[1, shape[1], shape[2], shape[3]])
		means = tf.tile(tf.reduce_mean(in_tensor, axis = [1, 2, 3])[:, tf.newaxis, tf.newaxis, tf.newaxis], 
			[1, shape[1], shape[2], shape[3]])
		
		new_tensor = in_tensor + (in_tensor - means) * factor
		return tf.clip_by_value(new_tensor, 0.0, 1.0)
	
	def _one_channel(in_tensor, f):
		
		shape = in_tensor.get_shape().as_list()
		factor = tf.tile(tf.random_uniform([shape[0]], f[0], f[1], dtype = tf.float32)[:, tf.newaxis, tf.newaxis],
			[1, shape[1], shape[2]])
		
		c = tf.random_uniform([shape[0]], 0, 3, dtype = tf.int32)
		c = tf.tile(c[:, tf.newaxis, tf.newaxis], [1, shape[1], shape[2]])

		channels = tf.unstack(in_tensor, shape[3], 3)
		new_channels = []
		for i, channel in enumerate(channels):
			new_channels.append(tf.where(tf.equal(i, c),
				channel + (channel - tf.tile(tf.reduce_mean(channel, axis = [1, 2])[:, tf.newaxis, tf.newaxis], 
					[1, shape[1], shape[2]]))*factor,
				channel))

		return tf.clip_by_value(tf.stack(new_channels, 3), 0.0, 1.0)

	def _no_norm(in_tensor, f):
		return in_tensor

	prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)

	per_channel_prob = 0.5

	return tf.case([(tf.less_equal(prob, 0.5*per_channel_prob), lambda: _normalize(in_tensor, f)),
		(tf.less_equal(prob, 0.5), lambda: _one_channel(in_tensor, f))],
		default = lambda: _no_norm(in_tensor, f))


@map_decorator('contrast_normalization')
def gamma_normalization(in_tensor, f = [0.5, 2.2]):
	'''
	Perform Gamma Contrast Normalization on the input tensor
	the normalization algorithm is the gamma normalization, 
	following the equation: I_N = I ** gamma
	where I is the image normalized between 0 and 1

	Inputs:
		in_tensor:	tensor to be normalized. Must
				either be 3D with format HxWxC or
				4D with format NxHxWxC (N - Batch size
				H - Height, W - Width, C - Channels)
		f:	normalization factor. normal range lies 
				between 0.5 and 2.0
	Returns:
		The normalized image batch
	'''
	def _normalize(in_tensor, f):

		shape = in_tensor.get_shape().as_list()
		gamma = tf.random_uniform([shape[0]], f[0], f[1], dtype = tf.float32)
		gamma = tf.tile(gamma[:, tf.newaxis, tf.newaxis, tf.newaxis], [1, shape[1], shape[2], shape[3]])
		new_tensor = tf.pow(in_tensor, gamma)#0.5 + alpha * (in_tensor - 0.5)

		return tf.clip_by_value(new_tensor, 0.0, 1.0)
	
	def _one_channel(in_tensor, f):
		
		shape = in_tensor.get_shape().as_list()
		gamma = tf.random_uniform([shape[0]], f[0], f[1], dtype = tf.float32)
		gamma = tf.tile(gamma[:, tf.newaxis, tf.newaxis], [1, shape[1], shape[2]])
		
		c = tf.random_uniform([shape[0]], 0, 3, dtype = tf.int32)
		c = tf.tile(c[:, tf.newaxis, tf.newaxis], [1, shape[1], shape[2]])

		channels = tf.unstack(in_tensor, shape[3], 3)
		new_channels = []
		for i, channel in enumerate(channels):
			new_channels.append(tf.where(tf.equal(i, c),
				tf.pow(channel, gamma),
				channel))

		return tf.clip_by_value(tf.stack(new_channels, 3), 0.0, 1.0)

	def _no_norm(in_tensor, f):
		return in_tensor

	prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)

	per_channel_prob = 0.3

	return tf.case([(tf.less_equal(prob, 0.5*per_channel_prob), lambda: _normalize(in_tensor, f)),
		(tf.less_equal(prob, 0.5), lambda: _one_channel(in_tensor, f))],
		default = lambda: _no_norm(in_tensor, f))

@map_decorator('multiply_brightness')
def multiply_brightness(in_tensor, factor_range = [0.6, 1.4]):
	'''
	Multiply the values of the tensor with a constant factor

	inputs:
		in_tensor:	Tensor to be augmented
		factor:	Multiplication factor
		per_channel:	gives a probability with which only
				one channel is augmented
	returns:
		the augmented tensor
	'''
	def _all_channels(in_tensor, factor):
		factor = tf.tile(factor[..., tf.newaxis], [1, 1, 1, in_tensor.get_shape().as_list()[3]])
		return tf.clip_by_value(in_tensor*factor, 0.0, 1.0)

	def _one_channel(in_tensor, factor):
		channel = tf.random_uniform([], minval = 0, maxval = 3, dtype = tf.int32)
		c = in_tensor.get_shape().as_list()[3]
		channels = tf.unstack(in_tensor, c, 3)
		new_channels = []
		for i, tensor_channel in enumerate(channels):
			new_channels.append(tf.where(tf.equal(i, channel),
				tensor_channel * factor,
				tensor_channel))

		return tf.clip_by_value(tf.stack(new_channels, 3), 0.0, 1.0)

	def _no_aug(in_tensor):
		return in_tensor

	prob = tf.random_uniform([], minval = 0.0, maxval = 1.0, dtype = tf.float32)
	
	shape = in_tensor.get_shape().as_list()
	factor = tf.random_uniform([shape[0]], minval = factor_range[0], maxval = factor_range[1], dtype = tf.float32)
	factor = tf.tile(factor[:, tf.newaxis, tf.newaxis], [1, shape[1], shape[2]])
	
	aug1 = lambda: _all_channels(in_tensor, factor)
	aug2 = lambda: _one_channel(in_tensor, factor)
	aug3 = lambda: _no_aug(in_tensor)

	return tf.case([(tf.less_equal(prob, 0.25), aug1), 
		(tf.less_equal(prob, 0.5), aug2)],
		default = aug3)

def main():
	#bg_img = tf.random_uniform([100, 64, 64, 1], 0.0, 1.0, dtype=tf.float32)
	in_image = tf.pad(tf.ones([10, 32, 32, 3], dtype = tf.float32)*0.5,
			paddings=[[0, 0 ], [16, 16], [16, 16], [0, 0]], mode = 'CONSTANT')+0.25
	#in_image = tf.pad(tf.ones([32, 32, 1], dtype = tf.float32),
	#		paddings=[[16, 16], [16, 16], [0, 0]], mode = 'CONSTANT')
	

	#out_tensor = zoom_image_object(in_image, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
	#out_tensor = translate_im_obj(in_image, [-10, 10])
	#out_tensor = zoom_and_translate(in_image, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], [-10, 10])
	
	#f = [0.6, 1.4]
	#shape = in_image.get_shape().as_list()
	#alpha = tf.random_uniform([in_image.get_shape().as_list()[0]], f[0], f[1], dtype = tf.float32)
	#alpha = tf.tile(alpha[:, tf.newaxis, tf.newaxis, tf.newaxis], [1, shape[1], shape[2], shape[3]])
	#new_tensor = 0.5 + alpha * (in_image - 0.5)
	#out_tensor = new_tensor
	out_tensor = contrast_normalization(in_image)
	# out_tensor = add_black_patches(in_image)

	

	shape = out_tensor.get_shape().as_list()


	with tf.Session() as sess:
		for i in range(10):
			#sess.run(tf.global_variables_initializer)
			inputs, results = sess.run([in_image, out_tensor])
			#print(results)
			print(shape)

			#print(a.shape)
			#print(vr_n)
			#print(factor)
			#print(test_sub)
			#print(test_square)
			#print(distance)
			
			#print(np.mean(inputs, axis=[1, 2, 3]))
			#print(np.mean(results, axis=[1, 2, 3]))
			#a_means = [np.mean(x) for x in a]
			in_means = [np.mean(x) for x in inputs]
			out_means = [np.mean(x) for x in results]
			#print(a_means)

			print(in_means)
			print(out_means)
			fig = plt.figure(figsize = (8, 8))
			r=2
			c=2
			
			if len(inputs.shape) == 4:
				fig.add_subplot(r, c, 1)
				plt.imshow(inputs[0, :, :, :])
				fig.add_subplot(r, c, 2)
				plt.imshow(results[0, :, :, :])
				fig.add_subplot(r, c, 3)
				plt.imshow(inputs[5, :, :, :])
				fig.add_subplot(r, c, 4)
				plt.imshow(results[5, :, :, :])
				#plt.savefig('test_figure_' + str(i) + '.png')
			else:
				fig.add_subplot(r, c, 1)
				plt.imshow(inputs[:, :, :])
				fig.add_subplot(r, c, 2)
				plt.imshow(results[:, :, :])
				#plt.savefig('test_figure_' + str(i) + '.png')

			plt.show()
			#plt.clf()
			#plt.close()
			#print(np.sum(results[15, :, :, 0]))
			# print(np.mean(inputs))
			# print(np.mean(results))

if __name__ == '__main__':
	main()
