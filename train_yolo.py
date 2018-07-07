import os,sys,caffe
import numpy as np

caffe.set_mode_gpu()

solver=caffe.SGDSolver(sys.argv[1])
if len(sys.argv)>2:
    solver.net.copy_from(sys.argv[2])
#solver.net.set_mode_gpu()

solver.solve()

print 'finished'