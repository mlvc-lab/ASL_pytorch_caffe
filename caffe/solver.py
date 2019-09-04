from __future__ import print_function
import os
import os.path as osp
import sys
from caffe.proto import caffe_pb2


class Solver:
    def __init__(self, solver_name=None, folder=None, 
            method='SGD', base_lr=0.1, lr_policy='multistep'):
        self.solver_name = solver_name
        self.folder = folder
        
        if self.folder is not None:
            self.name = osp.join(self.folder, 'solver.prototxt')
        if self.name is None:
            self.name = 'solver.pt'
        else:
            filepath, ext = osp.splitext(self.name)
            if ext == '':
                ext = '.prototxt'
                self.name = filepath+ext

        self.p = caffe_pb2.SolverParameter()        
        
        class Method:
            nesterov = "Nesterov"
            SGD = "SGD"
            AdaGrad = "AdaGrad"
            RMSProp = "RMSProp"
            AdaDelta = "AdaDelta"
            Adam = "Adam"

        self.method=Method()
        
        class Policy:
            """    - fixed: always return base_lr."""
            fixed = 'fixed'
            step = 'step'
            """    - step: return base_lr * gamma ^ (floor(iter / step))"""
            """    - exp: return base_lr * gamma ^ iter"""
            """    - inv: return base_lr * (1 + gamma * iter) ^ (- power)"""
            """    - multistep: similar to step but it allows non uniform steps defined by stepvalue"""
            multistep = 'multistep'
            """    - poly: the effective learning rate follows a polynomial decay, to be zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)"""
            """    - sigmoid: the effective learning rate follows a sigmod decay"""
            """      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))"""
        self.policy = Policy()

        class Machine:
            GPU = self.p.GPU
            CPU = self.p.GPU
        self.machine = Machine()

        # defaults

        self.p.test_iter.extend([100])
        self.p.test_interval = 1000
        self.p.test_initialization = True

        self.p.base_lr = base_lr 
        if lr_policy == 'multistep':
            self.p.lr_policy = self.policy.multistep
            self.p.stepvalue.extend([32000, 48000])
            self.p.gamma = 0.1
            self.p.max_iter = 64000

        elif lr_policy == 'step':
            self.p.lr_policy = self.policy.step
            self.p.stepsize = 10000
            self.p.gamma = 0.5
            self.p.max_iter = 80000 
        else:
            raise ValueError('no method !')

        self.p.momentum = 0.9
        self.p.weight_decay = 0.00004

        self.p.display = 1000
        self.p.snapshot = 10000
        self.p.snapshot_prefix = osp.join(self.folder, "snapshot/")
        self.p.solver_mode = self.machine.GPU

        if method == 'SGD':
            self.p.type = self.method.SGD
        elif method == 'Nesterov': 
            self.p.type = self.method.nesterov
        else:
            raise ValueError('no method !')

        self.p.net = osp.join(self.folder, "trainval.prototxt")

    def write(self):
        dirname = osp.dirname(self.name)
        if not osp.exists(dirname):
            os.mkdir(dirname)
        if not osp.exists(self.p.snapshot_prefix):
            os.mkdir(self.p.snapshot_prefix)
        with open(self.name, 'w') as f:
            f.write(str(self.p))
