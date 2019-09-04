import argparse
import os
import sys

import scipy.misc
import pickle
import numpy as np
import pdb

def make_image_data(root, dst):

    # Train set

    save_dir_train = os.path.join(dst, 'train')
    if not os.path.exists(save_dir_train):
        os.mkdir(save_dir_train)

    # Iterate on data_batch_(1 ~ 5)
    for _b in range(1,6):
        batch_filename = os.path.join(root, 'data_batch_{}'.format(_b))
        datadict = {}
        with open(batch_filename, 'rb') as f:
            d = pickle.load(f, encoding='bytes')
            for k, v in d.items():
                datadict[k.decode('utf8')] = v

        X = datadict['data'].reshape(10000, 3, 32, 32).astype(np.uint8)

        # Padding
        padded = np.zeros((10000, 3, 40, 40), dtype=np.uint8)
        padded[:,:,:,:] = 0 # zero padding 
        padded[:,:,4:-4, 4:-4] = X

        for i, filename in enumerate(datadict['filenames']):

            save_path = os.path.join(save_dir_train, filename.decode())
            img = padded[i].transpose(1,2,0)
            print(filename.decode())

            # Save image
            scipy.misc.toimage(img, cmin=0.0, cmax=255).save(save_path)

        # Write filepath, label in train.txt 
        with open(os.path.join(dst, 'train.txt'), 'a') as f:
            for filename, label in zip(datadict['filenames'], datadict['labels']):
                filepath = os.path.join(save_dir_train, filename.decode()) 
                f.write(filepath + ' ' + str(label) + '\n')

    # Test set

    save_dir_test = os.path.join(dst, 'test')
    if not os.path.exists(save_dir_test):
        os.mkdir(save_dir_test)

    batch_filename = os.path.join(root, 'test_batch')
    datadict = {}
    with open(batch_filename, 'rb') as f:
        d = pickle.load(f, encoding='bytes')
        for k, v in d.items():
            datadict[k.decode('utf8')] = v

    X = datadict['data'].reshape(10000, 3, 32, 32).astype(np.uint8)

#    # Padding
#    padded = np.zeros((10000, 3, 40, 40), dtype=np.uint8)
#    padded[:,:,:,:] = 0 # zero padding 
#    padded[:,:,4:-4, 4:-4] = X

    for i, filename in enumerate(datadict['filenames']):

        save_path = os.path.join(save_dir_test, filename.decode())
        img = X[i].transpose(1,2,0)
        print(filename.decode())

        # Save image
        scipy.misc.toimage(img, cmin=0.0, cmax=255).save(save_path)

    # Write filepath, label in test.txt 
    with open(os.path.join(dst, 'test.txt'), 'a') as f:
        for filename, label in zip(datadict['filenames'], datadict['labels']):
            filepath = os.path.join(save_dir_test, filename.decode()) 
            f.write(filepath + ' ' + str(label) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess the cifar10 data')
    parser.add_argument('root', type=str, default='cifar-10-batches-py',
                        help='path of dataset')
    parser.add_argument('--dst', type=str, default='cifar-10-batches-py',
                        help='destination of result')

    args = parser.parse_args()

    if not os.path.exists(args.dst):
        os.mkdir(args.dst)

    dst_train_path = os.path.join(args.dst, 'train')
    dst_test_path = os.path.join(args.dst, 'test')
    make_image_data(args.root, args.dst)

