import argparse
import os 
from subprocess import call

import numpy as np
from matplotlib import pylab as plt


def main(log_file, save_dir):

    parse_test(log_file)

    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('accuracy %')
    loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind = parse_log(log_file + '.test')
    disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, color_ind=0)
    #plt.show()
    filename = os.path.basename(log_file)
    save_path = os.path.join(save_dir, filename + '.plot.png') 
    plt.savefig(save_path, bbox_inches='tight')

def parse_test(log_file):
    log_dir = '/'.join(log_file.split('/')[:-1])
    call('/usr/bin/python {}/tools/extra/parse_log.py {} {}'
        .format(
            os.environ['CAFFE_ROOT'],
            os.path.abspath(log_file), 
            os.path.abspath(log_dir)), shell=True)

def parse_log(log_file):

    with open(log_file, 'r') as log_file:
        log = log_file.read().strip().split('\n')

    losses = []
    loss_iterations = []
    accuracies = []
    accuracy_iterations = []
    acc_iteration_checkpoints_ind = []

    for line in log[1:]:
        line_split = line.split(',') 
        num_iter = int(float(line_split[0]))
        if num_iter == 0:
            acc = 0
        else:
            acc = float(line_split[3]) * 100
        loss = float(line_split[4])
        losses.append(loss)
        accuracies.append(acc)
        loss_iterations.append(num_iter)
        accuracy_iterations.append(num_iter)
        if num_iter % 10000 == 0 and num_iter > 0:
            acc_iteration_checkpoints_ind.append(len(accuracy_iterations) - 1)
        
    loss_iterations = np.array(loss_iterations)
    losses = np.array(losses)

    accuracy_iterations = np.array(accuracy_iterations)
    accuracies = np.array(accuracies)
    
    print('index: ', np.argmax(accuracies), 'best acc: ', np.max(accuracies))

    return loss_iterations, losses, accuracy_iterations, accuracies, acc_iteration_checkpoints_ind


def disp_results(fig, ax1, ax2, loss_iterations, losses, accuracy_iterations, accuracies, accuracies_iteration_checkpoints_ind, color_ind=0):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
 
    modula = len(colors)
    ax1.plot(loss_iterations, losses, color=colors[(color_ind * 2 + 0) % modula])
    ax2.plot(accuracy_iterations, accuracies, color=colors[(color_ind * 2 + 1) % modula])
    ax2.plot(accuracy_iterations[accuracies_iteration_checkpoints_ind], accuracies[accuracies_iteration_checkpoints_ind], 'o', color=colors[(color_ind * 2 + 1) % modula])


if __name__ == '__main__':
     
    parser = argparse.ArgumentParser()
    parser.add_argument('in_file', type=str, help="path of log file in train log directory")
    parser.add_argument('out_dir', type=str, help="destination")
    args = parser.parse_args()
    main(args.in_file, args.out_dir)
