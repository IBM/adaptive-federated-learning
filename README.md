### Adaptive Federated Learning in Resource Constrained Edge Computing Systems

This repository includes source code for the paper S. Wang, T. Tuor, T. Salonidis, K. K. Leung, C. Makaya, T. He, and K. Chan, "Adaptive federated learning in resource constrained edge computing systems," IEEE Journal on Selected Areas in Communications, vol. 37, no. 6, pp. 1205 – 1221, Jun. 2019.

#### Getting Started

The code runs on Python 3 with Tensorflow version 1 (>=1.13). To install the dependencies, run
```
pip3 install -r requirements.txt
```

Then, download the datasets manually and put them into the `datasets` folder.
- For MNIST dataset, download from <http://yann.lecun.com/exdb/mnist/> and put the standalone files into `datasets/mnist`.
- For CIFAR-10 dataset, download the "CIFAR-10 binary version (suitable for C programs)" from <https://www.cs.toronto.edu/~kriz/cifar.html>, extract the standalone `*.bin` files and put them into `datasets/cifar-10-batches-bin`.

To test the code: 
- Run `server.py` and wait until you see `Waiting for incoming connections...` in the console output.
- Run 5 parallel instances of `client.py` on the same machine as the server. 
- You will see console outputs on both the server and clients indicating message exchanges. The code will run for a few minutes before finishing.
- After the server and clients finish, run `plot_multi_run.py` which will plot the result. This figure will look similar to the SVM(SGD) subfigures in Fig. 4 of the paper (but with higher fluctuation).

#### Code Structure

All configuration options are given in `config.py` which also explains the different setups that the code can run with.

The results are saved as CSV files in the `results` folder. 
The CSV files should be deleted before starting a new round of experiment.
Otherwise, the new results will be appended to the existing file.

Currently, the supported datasets are MNIST and CIFAR-10, and the supported models are SVM and CNN. The code can be extended to support other datasets and models too.  

#### Citation

When using this code for scientific publications, please kindly cite the above paper.

#### Contributors

This code was written by Shiqiang Wang and Tiffany Tuor.
