import sys

sys.path.append('./python')
sys.path.append('./python/caffe')

sys.path.append('/home/sensetime/DeepLearning/test/fcn-pool')


import caffe
import surgery, score

import numpy as np
import os

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../fcn-pool/snapshot/fcn8s-heavy-pascal.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('../fcn-pool/fcn-improve/solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../fcn-pool/data/fcn-gf2-512-1024/valdata/val.txt', dtype=str)

for _ in range(10):
    solver.step(2000)
    score.seg_tests(solver, False, val, layer='score')
