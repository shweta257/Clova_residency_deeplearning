This repository is a Tensorflow implementation of CapsNet based on Geoffrey Hinton's paper [Dynamic Routing Between Capsules](https://arxiv.org/abs/1710.09829)

## Requirements
- Python
- NumPy
- [Tensorflow](https://github.com/tensorflow/tensorflow) (I'm using 1.3.0, not yet tested for older version)
- tqdm (for displaying training progress info)

To run this code:
1. Clone the repository: git clone https://github.com/shweta257/Clova_residency_deeplearning.git
2. Download data (MNIST dataset)
	mkdir -p data/mnist
	wget -c -P data/mnist http://yann.lecun.com/exdb/mnist/{train-images-idx3-ubyte.gz,train-labels-idx1-ubyte.gz,t10k-images-idx3-ubyte.gz,t10k-labels-idx1-ubyte.gz}
	gunzip data/mnist/*.gz
3. Train the model:
	python clova_dynamic_routing.py

NOTE: To write this code I have referred another implementation of same paper on github (https://github.com/naturomics/CapsNet-Tensorflow.git)
