from skimage import io
import os
import numpy as np


path1='/home/sensetime/DeepLearning/test/fcn.berkeleyvision.org/data/fcn-gf2-512-1024/traindata/img'
str=os.listdir(path1)
l=len(str)

d1=0;
d2=0;
d3=0;
i=0
for x in str:
    i=i+1
    im=io.imread(path1+"/"+x)
    if i%1000==0:
        print i
    d1=d1+np.mean(im[:,:,0])
    d2=d2+np.mean(im[:,:,1])
    d3=d3+np.mean(im[:,:,2])

print  [d1/l,d2/l,d3/l]
print l

# fcn_zy3 train RGB [89.611119269769276, 105.91758338559417, 109.3944624054897]

# g=fcn zy3 512-1024  rgb [89.279196311370441, 105.57667462140543, 109.08637249794963]