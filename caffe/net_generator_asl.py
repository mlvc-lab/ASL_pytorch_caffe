from __future__ import print_function
import argparse
import os.path as osp
import sys
import os

from caffe.proto import caffe_pb2
from solver import Solver


class Net:
    def __init__(self, name="network"):
        self.net = caffe_pb2.NetParameter()
        self.net.name = name
        self.bottom = None
        self.cur = None
        self.this = None
    
    def setup(self, name, layer_type, bottom=[], top=[], inplace=False):

        self.bottom = self.cur

        new_layer = self.net.layer.add()

        new_layer.name = name
        new_layer.type = layer_type

        if self.bottom is not None and new_layer.type != 'Data' \
            and new_layer.type != 'ImageData':
            bottom_name = [self.bottom.name]
            if len(bottom) == 0:
                bottom = bottom_name
            new_layer.bottom.extend(bottom)
        
        if inplace:
            top = bottom_name
        elif len(top) == 0:
            top = [name]
        new_layer.top.extend(top)

        self.this = new_layer
        if not inplace:
            self.cur = new_layer

    def suffix(self, name, self_name=None):
        if self_name is None:
            return self.cur.name + '_' + name
        else:
            return self_name

    def write(self, name=None, folder=None):
        # dirname = osp.dirname(name)
        # if not osp.exists(dirname):
        #     os.mkdir(dirname)
        if folder is not None:
            name = osp.join(folder, 'trainval.prototxt')
        elif name is None:
            name = 'trainval.pt'
        else:
            filepath, ext = osp.splitext(name)
            if ext == '':
                ext = '.prototxt'
                name = filepath+ext
        with open(name, 'w') as f:
            f.write(str(self.net))

    def show(self):
        print(self.net)
    #************************** params **************************

    def param(self, lr_mult=1, decay_mult=0):
        new_param = self.this.param.add()
        new_param.lr_mult = lr_mult
        new_param.decay_mult = decay_mult

    def transform_param(self, 
            mean_value=128, 
            batch_size=128, 
            scale=0.0078125, 
            mirror=1, crop_size=None, mean_file_size=None, phase=None):

        new_transform_param = self.this.transform_param
        if scale != 1.:
            new_transform_param.scale = scale
        new_transform_param.mean_value.extend([mean_value])
        if phase is not None and phase == 'TEST':
            return

        new_transform_param.mirror = mirror
        if crop_size is not None:
            new_transform_param.crop_size = crop_size
        
    def data_param(self, source, backend='LMDB', batch_size=128):
        if backend == 'LMDB':
            new_data_param = self.this.data_param
            new_data_param.source = source
            new_data_param.backend = new_data_param.LMDB
            new_data_param.batch_size = batch_size    
        else:
            NotImplementedError
    
    def weight_filler(self, filler='msra'):
        """xavier"""
        if self.this.type == 'InnerProduct':
            self.this.inner_product_param.weight_filler.type = 'xavier'
        else:
            self.this.convolution_param.weight_filler.type = filler
    
    def bias_filler(self, filler='constant', value=0):
        if self.this.type == 'InnerProduct':
            self.this.inner_product_param.bias_filler.type = filler
            self.this.inner_product_param.bias_filler.value = value
        else:
            self.this.convolution_param.bias_filler.type = filler
            self.this.convolution_param.bias_filler.value = value

    def include(self, phase='TRAIN'):
        if phase is not None:
            includes = self.this.include.add()
            if phase == 'TRAIN':
                includes.phase = caffe_pb2.TRAIN
            elif phase == 'TEST':
                includes.phase = caffe_pb2.TEST
        else:
            NotImplementedError


    #************************** inplace **************************

    def ReLU(self, name=None, bottom=[], inplace=False):
#        self.setup(self.suffix('relu', name), 'ReLU', inplace=True)
        self.setup(name, 'ReLU', bottom=bottom, inplace=inplace)
    
    def BatchNorm(self, name=None, bottom=[], phase='TRAIN', inplace=False):
#        self.setup(self.suffix('bn', name), 'BatchNorm', inplace=True)
        self.setup(name, 'BatchNorm', bottom=bottom, inplace=inplace)

        self.param(lr_mult=0, decay_mult=0)
        self.param(lr_mult=0, decay_mult=0)
        self.param(lr_mult=0, decay_mult=0)
        batch_norm_param = self.this.batch_norm_param
        batch_norm_param.moving_average_fraction = 0.9

#        includes = self.this.include.add()
#        if phase == 'TRAIN':
#            batch_norm_param.use_global_stats = False
#            includes.phase = caffe_pb2.TRAIN
#        elif phase == 'TEST':
#            batch_norm_param.use_global_stats = True 
#            includes.phase = caffe_pb2.TEST
#        else:
#            raise ValueError('[!] phase is wrong.')

    def Scale(self, name=None, bottom=[], inplace=False):
#        self.setup(self.suffix('scale', name), 'Scale', inplace=)
        self.setup(name, 'Scale', bottom=bottom, inplace=inplace)
        self.param(lr_mult=1, decay_mult=0)
        self.param(lr_mult=1, decay_mult=0)
        self.this.scale_param.bias_term = True

    #************************** layers **************************

    def Data(self, source, top=['data', 'label'], name="data", phase=None,
            batch_size=128, **kwargs):
        self.setup(name, 'Data', top=top)

        self.include(phase)

        self.data_param(source, batch_size=batch_size)
        self.transform_param(phase=phase, **kwargs)

    def ImageData(self, source, name='data',
            top=['data', 'label'], phase=None,
            batch_size=128, shuffle=False, **kwargs):

        self.setup(name, 'ImageData', top=top)
        self.include(phase)

#        self.image_data_param(source, batch_size=batch_size)
        new_image_data_param = self.this.image_data_param
        new_image_data_param.source = source
        new_image_data_param.batch_size = batch_size    
        new_image_data_param.shuffle = shuffle 

        self.transform_param(phase=phase, **kwargs)

    def ActiveShift(self, name=None, bottom=[],
            pad=0, stride=1,
            decay=False, freeze=False):
        self.setup(name, 'ActiveShift',  bottom=bottom)
        asl_param = self.this.asl_param

        asl_param.pad = pad
        asl_param.stride = stride
        asl_param.normalize = True
        self.param(lr_mult=0.001, decay_mult=0.)
        self.param(lr_mult=0.001, decay_mult=0.)
        
        asl_param.shift_filler.type = 'uniform'
        asl_param.shift_filler.min = -1.
        asl_param.shift_filler.max = 1.
        
    def Convolution(self, name, 
            bottom=[], num_output=None, 
            kernel_size=3, pad=1, stride=1, group=1,
            decay = True, bias = False, freeze = False):
        self.setup(name, 'Convolution', bottom=bottom, top=[name])
        
        conv_param = self.this.convolution_param
#        if num_output is None:
#            num_output = self.bottom.convolution_param.num_output
        assert num_output is not None, '[!] num_output is None.'

        conv_param.num_output = num_output
        conv_param.pad.extend([pad])
        conv_param.kernel_size.extend([kernel_size])
        conv_param.stride.extend([stride])
        conv_param.group = group
        
        if freeze:
            lr_mult = 0
        else:
            lr_mult = 1
        if decay:
            decay_mult = 1
        else:
            decay_mult = 0
        self.param(lr_mult=lr_mult, decay_mult=decay_mult)
        self.weight_filler()

        if bias:
            #if decay:
            #    decay_mult = 2
            #else:
            decay_mult = 0
            self.param(lr_mult=2*lr_mult, decay_mult=decay_mult)
            self.bias_filler()
        else:
            conv_param.bias_term = False

            
        
    def SoftmaxWithLoss(self, name='loss', label='label'):
        self.setup(name, 'SoftmaxWithLoss', bottom=[self.cur.name, label])

    def Softmax(self,bottom=[], name='softmax'):
        self.setup(name, 'Softmax', bottom=bottom)

    def Accuracy(self, name='Accuracy', label='label'):
        self.setup(name, 'Accuracy', bottom=[self.cur.name, label])


    def InnerProduct(self, name='fc', num_output=10):
        self.setup(name, 'InnerProduct')
        self.param(lr_mult=1, decay_mult=1)
        self.param(lr_mult=1, decay_mult=0)    
        inner_product_param = self.this.inner_product_param
        inner_product_param.num_output = num_output
        self.weight_filler()
        self.bias_filler()
    
    def Pooling(self, name, bottom=[], pool='AVE', global_pooling=False):
        """MAX AVE """
        self.setup(name, 'Pooling', bottom=bottom)
        if pool == 'AVE':
            self.this.pooling_param.pool = self.this.pooling_param.AVE
        else:
            NotImplementedError
        if global_pooling:
            self.this.pooling_param.global_pooling = global_pooling
        else:
            self.this.pooling_param.pad = 1
            self.this.pooling_param.kernel_size = 3
            self.this.pooling_param.stride = 2

    def Eltwise(self, name, bottom0, bottom1, operation='SUM'):
        print('bottom0', bottom0)
        print('bottom1', bottom1)
        self.setup(name, 'Eltwise', bottom=[bottom0, bottom1])
        if operation == 'SUM':
            self.this.eltwise_param.operation = self.this.eltwise_param.SUM
        else:
            NotImplementedError


    #************************** DIY **************************
    def conv_relu(self, name, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.ReLU(relu_name)

    def conv_bn_relu(self, name, bn_name=None, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(None)
        self.ReLU(relu_name)

    def conv_bn(self, name, bn_name=None, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(None)

    def bn_relu_3x3dw(self, name, bn_name=None, relu_name=None, **kwargs):
        self.Convolution(name, **kwargs)
        self.BatchNorm(bn_name)
        self.Scale(None)
        self.ReLU(relu_name)

    def softmax_acc(self,bottom, **kwargs):
        self.Softmax(bottom=[bottom])

        has_label=None
        for name, value in kwargs.items():
            if name == 'label':
                has_label = value
        if has_label is None:
            self.Accuracy()
        else:
            self.Accuracy(label=has_label)
            

    #************************** network blocks **************************

    def res_func(self, name, num_input, num_output, up=False):
        bottom = self.cur.name
        print('name:', name)
        print('bottom:', bottom)
        self.BatchNorm(name + '_bn0', bottom=[bottom], phase='TRAIN')
#        self.BatchNorm(name + '_bn0', bottom=[bottom], phase='TEST')
        self.Scale(name + '_scale0', bottom=[name + '_bn0'], inplace=True)
        self.ReLU(name + '_relu0', bottom=[name + '_bn0'], inplace=True)
        self.Convolution(name + '_conv0', bottom=[name + '_bn0'],
            num_output=num_output, kernel_size=1, pad=0, stride=1,
            group=1, bias=False)

        self.BatchNorm(name + '_bn1', bottom=[name + '_conv0'], phase='TRAIN', inplace=True)
#        self.BatchNorm(name + '_bn1', bottom=[name + '_conv0'], phase='TEST')
        self.Scale(name + '_scale1', bottom=[name + '_conv0'], inplace=True)
        self.ReLU(name + '_relu1', bottom=[name + '_conv0'], inplace=True)
        self.ActiveShift(name + '_asl', bottom=[name + '_conv0'], stride=1+int(up))
        self.Convolution(name + '_conv1', bottom=[name + '_asl'],
            num_output=num_output, kernel_size=1, pad=0, stride=1,
            group=1, bias=False)

        if up:
            self.Convolution(name + '_proj', bottom=[name + '_bn0'],
                num_output=num_output, kernel_size=1, pad=0, stride=2,
                group=1, bias=False)
#            self.Pooling(name + '_proj', bottom=[bottom], pool='AVE', global_pooling=False)
            self.Eltwise(name + '_sum', bottom0=(name + '_proj'), bottom1=(name + '_conv1'))
        else:
            self.Eltwise(name + '_sum', bottom0=bottom, bottom1=(name + '_conv1'))
    
    def res_group(self, group_id, n, num_input, num_output):
        def name(block_id):
            return 'group{}'.format(group_id) + '_block{}'.format(block_id)

        if group_id == 0:
            up = False
        else:
            up = True
        self.res_func(name(0), num_input, num_output, up=up)
        for i in range(1, n):
            self.res_func(name(i), num_output, num_output)

    #************************** networks **************************
        
    def asl_resnet_cifar(self, n=3, base_width=16):
        """6n+2, n=3 9 18 coresponds to 20 56 110 layers
        first 3x3conv -> (ch 16, size 32)
        1st block = bn - relu - 1x1conv - bn - relu - asl- 1x1conv -> (ch 16, size 32)
        2nd block -> (ch 32, size 16)
        3rd block -> (ch 64, size 8)
        global average pooling
        fc

        """
        num_output = base_width
        self.Convolution('first_conv', 
            num_output=num_output, kernel_size=3, pad=1, stride=1, bias=False)
        for i in range(3):
            if i == 0:
                self.res_group(i, n, num_output, num_output*(2**i))
            else:
                self.res_group(i, n, num_output*(2**(i-1)), num_output*(2**i))
        
        self.BatchNorm('last_bn', bottom=['group2_block2_sum'], phase='TRAIN', inplace=True)
#        self.BatchNorm('last_bn', bottom=['group2_block2_sum'], phase='TEST', inplace=True)
        self.Scale('last_scale', bottom=['group2_block2_sum'], inplace=True)
        self.ReLU('last_relu', bottom=['group2_block2_sum'], inplace=True)
        self.Pooling("global_avg_pool", bottom=['group2_block2_sum'], global_pooling=True)
        self.InnerProduct()
        self.SoftmaxWithLoss()
        self.softmax_acc(bottom='fc')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, help="name of model", required=True)
    parser.add_argument('--lr', type=float, help="learning rate", required=True)
    parser.add_argument('--bw', type=int, help="base width", required=True)
    args = parser.parse_args()

    #3, 5, 7, 9, 18
    n = 3

    solver = Solver(folder=args.name, method='Nesterov', base_lr=args.lr)
    solver.write()

    builder = Net(args.name)
    builder.ImageData('data_img/train.txt', phase='TRAIN', batch_size=128, shuffle=True, crop_size=32)
    builder.ImageData('data_img/test.txt', phase='TEST', batch_size=100, shuffle=False)
    builder.asl_resnet_cifar(n, args.bw)
    builder.write(folder=args.name)

