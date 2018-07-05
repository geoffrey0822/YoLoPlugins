import os,sys,caffe
import numpy as np

caffe.set_mode_gpu()

solver=caffe.SGDSolver(sys.argv[1])
#solver.net.set_mode_gpu()

solver.solve()

print 'finished'