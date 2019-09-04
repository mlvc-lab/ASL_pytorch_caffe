cifar10="cifar-10-python.tar.gz"
mkdir data
cd data
wget https://www.cs.toronto.edu/~kriz/$cifar10
tar xzvf $cifar10
rm $cifar10

cifar100="cifar-100-python.tar.gz"
wget https://www.cs.toronto.edu/~kriz/$cifar100
tar xzvf $cifar100
rm $cifar100
