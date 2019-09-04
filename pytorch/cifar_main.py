'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv

from utils import progress_bar

#from test_models import *
#from test_models import test_asl_module
from models import *
from models import asl_module

from logger import Logger

#from vis_shift_param import plot_main

#from cls import CyclicLR

import pdb


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--name', type=str)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--arch', type=str, help='model architecture')
parser.add_argument('--lr', default=0.04, type=float, help='learning rate')
parser.add_argument('--asl_lr', default=2e-2, type=float, help='asl learning rate')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
parser.add_argument('--asl_weight_decay', default=0, type=float, help='weight decay')
parser.add_argument('--nesterov', action='store_true', default=False, help='nesterov option in SGD')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--test', action='store_true', default=False, help='test trained model.')
parser.add_argument('--base_width', default=88, type=int, help='base width.')
args = parser.parse_args()


log_dir = './logs/{}'.format(args.name)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
logger = Logger(log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 160

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar10':
    dataloader = torchvision.datasets.CIFAR10
    num_classes = 10
elif args.dataset == 'cifar100':
    dataloader = torchvision.datasets.CIFAR100
    num_classes = 100
else:
    raise NotImplementedError('choose [ cifar10 / cifar100 ]')

trainset = dataloader(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = dataloader(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

if args.arch == 'asl':
    net = asl_resnet18(args.base_width, num_classes=num_classes)
elif args.arch == 'hgc':
    net = hgc_asl_resnet18(args.base_width, num_classes=num_classes)
elif args.arch == 'shuffle':
    net = shuffle_asl_resnet18(args.base_width, num_classes=num_classes)
elif args.arch == 'few_asl':
    net = few_asl_resnet18(args.base_width, num_classes=num_classes)
elif args.arch == 'skip_asl':
    net = skip_asl_resnet18(args.base_width, num_classes=num_classes)
elif args.arch == 'resnet':
    net = resnet18(args.base_width, num_classes=num_classes)
else:
    raise NotImplementedError('No arch: {}'.format(args.arch))

# net = VGG('VGG19')
#net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}_ckpt.t7'.format(args.name))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

params = []
#layer_count = 1
#for child in net.module.children():
#    #print(child)
#    
#    if isinstance(child, torch.nn.modules.conv.Conv2d):
#        #print('conv')
#        params.append({'params': child.parameters()})
#    elif isinstance(child, torch.nn.modules.container.Sequential):
#        #print('layer_{}'.format(layer_count))
#        layer_count += 1
#        block_count = 1
#        for block in child.children():
#            #print('\tblock_{}'.format(block_count))
#            block_count += 1
#            for layer in block.children():
#                if isinstance(layer, torch.nn.modules.batchnorm.BatchNorm2d):
#                    #print('\t\tbn')
#                    params.append({'params': layer.parameters()})
#                elif isinstance(layer, asl_module.ActiveShiftLayer):
#                    #print('\t\tasl')
#                    params.append({'params': layer.parameters(), 'lr': args.asl_lr, 'weight_decay': args.asl_weight_decay})
#                elif isinstance(layer, torch.nn.modules.conv.Conv2d):
#                    #print('\t\tconv')
#                    params.append({'params': layer.parameters()})
#                elif isinstance(layer, torch.nn.modules.container.Sequential):
#                    #print('\t\tshortcut')
#                    params.append({'params': layer.parameters()})
#                    
#    elif isinstance(child, torch.nn.modules.batchnorm.BatchNorm2d):
#        #print('bn')
#        params.append({'params': child.parameters()})
#    elif isinstance(child, torch.nn.modules.linear.Linear):
#        #print('linear')
#        params.append({'params': child.parameters()})
#    else:
#        ValueError('[!] Error in net.')

for m in net.module.modules():
    if isinstance(m, torch.nn.modules.conv.Conv2d):
        params.append({'params': m.parameters()})
    elif isinstance(m, torch.nn.modules.batchnorm.BatchNorm2d):
        params.append({'params': m.parameters()})
    elif isinstance(m, asl_module.ActiveShiftLayer):
        params.append({'params': m.parameters(), 'lr': args.asl_lr, 'weight_decay': args.asl_weight_decay})
    elif isinstance(m, torch.nn.modules.linear.Linear):
        params.append({'params': m.parameters()})
#    elif isinstance(m, torch.nn.modules.container.Sequential):
#        pass
    else:
        print(type(m))
        #raise ValueError('[!] Error in net.')

#optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1) 
#scheduler = CyclicLR(optimizer, base_lr=1e-3, max_lr=0.037, step_size=4000)
# total: 64000 iter, lr_decay: [32000 iter, 48000 iter]

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        #scheduler.batch_step()

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Logging in tensorboard
    info = {'train_loss': train_loss/(batch_idx+1), 'train_acc': 100.*correct/total}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch+1)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Logging in tensorboard
    info = {'val_loss': test_loss/(batch_idx+1), 'val_acc': 100.*correct/total}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch+1)


    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}_ckpt.t7'.format(args.name))
        best_acc = acc

def adjust_learning_rate(epoch):
#    # Find min_lr, max_lr
#    for i in range(100):
#        if epoch == i:
#            for param_group in optimizer.param_groups:
#                param_group['lr'] = ((0.037 - 0.001) / 9) * i + 0.001
#            break
#    return

    if epoch < 80:
        return
    elif 80 <= epoch < 120:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * 0.1
    elif epoch >= 120:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * 0.1 * 0.1

if args.test:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}_ckpt.t7'.format(args.name))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded {}...'.format(args.name))
    print('best_acc: ', best_acc)
    test(0)
    exit()



for epoch in range(start_epoch, total_epoch):

    lr1 = 0
    lr2 = 0
    for param_group in optimizer.param_groups:
        if param_group['initial_lr'] == args.lr and lr1 == 0:
            lr1 = param_group['lr']
        elif param_group['initial_lr'] == args.asl_lr:
            lr2 = param_group['lr']
            break

    print('epoch {}, lr1: {}, lr2: {}'.format(epoch, lr1, lr2))

#    # Find min_lr, max_lr
#    adjust_learning_rate(epoch)
#    print(scheduler.get_lr())

#    print(scheduler.get_lr())
    train(epoch)
    test(epoch)
    adjust_learning_rate(epoch)

#    scheduler.step()

    # Plot shift param
    #plot_main(net, args.name, 'pngs', epoch)


with open('exp_result.csv', 'a', encoding='utf-8', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow([args.name, best_acc])
     
