# asl_hgc_project

creator: jbjeong  


### 0. Prerequisite

Install active shift layer and use **asl_cuda** like python package.  
Note that you have to build the asl package.  
In this example, we use virtualenv.
  
We assume that this environment is set.
* python3.6 
* pytorch-v1.1 
* cuda 10.0  
<pre>
<code>$ pip install -r requirements.txt</code>
</pre>

First, install pytorch source from github.
<pre>
<code>$ cd ~
$ git clone https://github.com/jbjeong/asl_hgc_project.git
$ cd ~
$ git clone https://github.com/pytorch/pytorch
$ cd pytorch
$ cp -r ~/asl_hgc_project/pytorch/active_shift_layer_cuda .
$ cd active_shift_layer_cuda
$ python setup.py install</code>
</pre>

Test installation.
<pre>
<code>$ python -c "import torch; import asl_cuda" </code>
</pre>

### 1. Prepare data
<pre>
<code>$ cd ~/asl_hgc_project/pytorch
$ ./download_cifar.sh</code>
</pre>
This makes **data** directory.

### 2. Train on cifar data.
<pre>
<code>$ python cifar_main.py --name test-asl-net --dataset cifar10 --arch asl --nesterov --base_width 16</code>
</pre>

### 2. Train on imagenet data.
<pre>
<code>$ python imagenet_main.py --name test-hgc-net --data /data --arch hgc --nesterov --base_width 18</code>
</pre>

