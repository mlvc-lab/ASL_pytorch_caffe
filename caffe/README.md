# asl_hgc_project

creator: jbjeong  


### 0. Prerequisite

Install active shift layer from https://github.com/jyh2986/Active-Shift  
Note that you have to build total caffe source whom the active shift layer is added to.
<pre>
<code>$ git clone https://github.com/jyh2986/Active-Shift
$ cd Active-Shift</code>
</pre>
Modify Makefile.config and install active shift layer with reference to https://caffe.berkeleyvision.org/installation.html
  
After installation, You can test the installation like below. If the installation is successful, this would make no error.
<pre>
<code>$ python -c "import caffe" </code>
</pre>

### 1. Prepare data
<pre>
<code>$ ./download_cifar.sh
$ python make_cifar10_image_data.py</code>
</pre>
This makes **cifar-10-batches-py** directory and **data_img** directory.

### 2. Make model proto file
<pre>
<code>$ python net_generator_asl.py --name test-asl-net --lr 0.1 --bw 16
</code>
</pre>
This makes **test-asl-net** directory.

### 2. Train
Use train.sh for training the network.
Usage: ./train.sh {GPU index} {model_name}
<pre>
<code>$ ./train.sh 0,1,2,3 test-asl-net</code>
</pre>

### 3. Usage of asl-hgc(active shift layer + heterogenous grouped convolution)
Similarly, you can use asl-hgc
<pre>
<code>$ python net_generator_asl.py --name test-hgc-net --lr 0.1 --bw 18
./train.sh 0 test-hgc-net</code>
</pre>


