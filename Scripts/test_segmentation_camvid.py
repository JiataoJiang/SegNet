import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import math
import pylab
from sklearn.preprocessing import normalize
caffe_root = 'caffe-segnet/' 			# Change this to the absolute directoy to SegNet Caffe
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)


for i in range(0, args.iter):

	net.forward()

	image = net.blobs['data'].data
	label = net.blobs['label'].data
	predicted = net.blobs['prob'].data
	image = np.squeeze(image[0,:,:,:])
	output = np.squeeze(predicted[0,:,:,:])
	ind = np.argmax(output, axis=0)

	r = ind.copy()
	g = ind.copy()
	b = ind.copy()
	r_gt = label.copy()
	g_gt = label.copy()
	b_gt = label.copy()

	Road = [255,0,0]

	rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb[:,:,0] = 255
	rgb[:,:,1] = 0
	rgb[:,:,2] = b*255
	rgb_gt = np.zeros((ind.shape[0], ind.shape[1], 3))
	rgb_gt[:,:,0] = 255
	rgb_gt[:,:,1] = 0
	rgb_gt[:,:,2] = b_gt*255


	image = np.transpose(image, (1,2,0))
	output = np.transpose(output, (1,2,0))
	image = image[:,:,(2,1,0)]


	#scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save(IMAGE_FILE+'_segnet.png')

	#scipy.misc.toimage(image, cmin=0.0, cmax=255).save('image'+'_segnet.png')
	#scipy.misc.toimage(rgb, cmin=0.0, cmax=255).save('test/'+str(i)+'.png')
	scipy.misc.toimage(rgb_gt[:,:,2], cmin=0.0, cmax=255).save('test/'+str(i)+'_gt_255'+'.png')
	scipy.misc.toimage(rgb[:,:,2], cmin=0.0, cmax=255).save('test/'+str(i)+'_255'+'.png')
	#scipy.misc.toimage(rgb_gt, cmin=0.0, cmax=255).save('rgb_gt'+'_segnet.png')


	#plt.figure()
	#plt.imshow(image,vmin=0, vmax=1)
	#plt.figure()
	#plt.imshow(rgb_gt,vmin=0, vmax=1)
	#plt.figure()
	#plt.imshow(rgb,vmin=0, vmax=1)
	#plt.show()


print 'Success!'

