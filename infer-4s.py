import numpy as np
from PIL import Image
import sys
from skimage.io import imread ,imshow ,imsave
from skimage import img_as_ubyte
from skimage import util

from copy import deepcopy


sys.path.append('/home/sensetime/DeepLearning/test/caffe/python')
sys.path.append('/home/sensetime/DeepLearning/test/caffe/python/caffe')

import caffe
import matplotlib.pyplot as plt

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('../imgtest/460.jpg')
#im = imread('/home/sensetime/DeepLearning/fxwaterextract/data/water/1024/121.tif')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((109.39446,105.91758,89.61112))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('../fcn-pool/fcn-improve/deploy.prototxt', '../fcn-pool/fcn-improve/snapshot/train_iter_15000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['score'].data[0].argmax(axis=0)
print type(out),out.dtype,out.shape,out.size

imsave('../fcn-pool/data/fcn-gf2-512-1024/fcn-improve/fcn4s-460-1.5w.png',out)

out1 = net.blobs['softmax_score'].data[0]
print type(out1),out1.dtype,out1.shape,out1.size
#dimage=img_as_ubyte(out)

#img=Image.open(out);
#imshow(out)
#show()
for x in range(1,11,1):
    im_1 = img_as_ubyte(out1[x])
    imsave("../fcn-pool/data/fcn-gf2-512-1024/fcn4s-improve/"+str(x)+"-460-1.5w-fcn.png",im_1)

print "success"
# img.save('test3.png','png')
# plt.imshow(out);
# plt.axis('off')
# plt.savefig('test2.png')
# plt.show()
