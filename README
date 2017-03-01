Bitfusion Ubuntu 14 Tensorflow AMI
==============================================================================



#### Contact Us

* [Join us on Slack](https://slack-bitfusion-aws.herokuapp.com/)  [![](https://slack.global.ssl.fastly.net/272a/img/icons/favicon-16.png)](https://slack-bitfusion-aws.herokuapp.com/)

* [Send us a note](http://www.bitfusion.io/support/)                                                                                                                                      





Getting started - Launch the AMI
-------------------------------------------------------------------------------

Subscribe to and launch the AMI from here:

[Launch the AMI](https://aws.amazon.com/marketplace/pp/B01EYKBEQ0)


EC2 Instance Access
-------------------------------------------------------------------------------

To get started, launch an AWS instances using this AMI from the EC2
Console. If you are not familiar with this process please review the AWS
documentation provided here:

http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/launching-instance.html

Accessing the instance via SSH:

```
ssh -i <path to your pem file> ubuntu@{ EC2 Instance Public IP }
```

Jupyter Notebook - http://{ EC2 Instance Public IP }:8888
-------------------------------------------------------------------------------

#### Logging In

You can login to the notebook at:

  * `http://{EC2 Instance Public IP}:8888`
  * The login PASSWORD is set to the Instance ID.

You can get the Instance ID (Jupyter Notebook Password) from the EC2 console by
clicking on the running instance, or if you are logged in via ssh you can obtain
it by executing the following command:

```
  ec2metadata --instance-id
```


#### Updating the HASHED Jupyter Login Password:

**It is highly recommended that you change the Jupyter login password.**

When logged in via ssh you can update the hashed password using one of the following functions:

 * for iPython 5 ```IPython.lib.passwd```
 * for any earlier release of iPython ```notebook.auth.security.passwd()```:

iPython 5 example:
```
  ipython
  In [1]: from IPython.lib import passwd
  In [2]: passwd()
  Enter password:
  Verify password:
  Out[2]: 'sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed'
  exit()
```

iPython 4 and earlier example:
```
  ipython
  In [1]: from notebook.auth import passwd
  In [2]: passwd()
  Enter password:
  Verify password:
  Out[2]: 'sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed'
  exit()
```

You can then add the outputed hashed password, which should look similar to ```sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed```
,to your Jupyter config file. The default location for this file is ~/.jupyter/jupyter_notebook_config.py. If you scroll
to the bottom of the file you will see the configuration entry that needs to be updated:

Example:

```
  c.NotebookApp.password = u'sha1:67c9e60bb8b6:9ffede0825894254b2e042ea597d771089e11aed'
```

Place your update string in place the old one and restart Jupyter to have the password take effect.

```
  sudo service jupyter restart
```


#### Notebook Location

The default notebook directory is /home/ubuntu/pynb.  This directory is
required for Jupyter to function.  If it is deleted you will need
recreate it and ensure it is owned by the ubuntu user.

TensorFlow
-------------------------------------------------------------------------------

TensorFlow and examples projects are installed within the ubuntu users home
directory (i.e /home/ubuntu).

This AMI has tensorflow installed with AWS GPU support. If a TensorFlow
operation has both CPU and GPU implementations, the GPU devices will be given
priority when the operation is assigned to a device. For example, matmul has
both CPU and GPU kernels. On a system with devices cpu:0 and gpu:0, gpu:0 will
be selected to run matmul.

When you run this AMI on a non-GPU instance you may see the following erros and
warning when running TensorFlow. These can be safely ignored, they simply indicate
that no GPU device was found on the system:
```
  E tensorflow/stream_executor/cuda/cuda_driver.cc:491] failed call to cuInit: CUDA_ERROR_NO_DEVICE
  I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:140] kernel driver does not appear to be running on this host (ip-###-##-##-###): /proc/driver/nvidia/version does not exist
  I tensorflow/core/common_runtime/gpu/gpu_init.cc:81] No GPU devices available on machine.
```

#### TensorFlow Notebook Tutorials

The assignments for the Udacity Deep Learning class with TensorFlow have been
added the Jupyter notebook under the directory udacity.

  http://{ EC2 Instance Public IP }:8888/tree/udacity#

Course information can be found at:

  https://www.udacity.com/course/deep-learning--ud730

Github README can be found at:

  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity



#### Tensorflow CLI Examples

The TensorFlow examples can be executed either using python to use python 2 or
using the python3 command to utilize python 3.

##### Example 1: Import Python Module

```
  $ python
    >>> import tensorflow as tf
    I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcublas.so.7.5 locally
    I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcudnn.so.4 locally
    I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcufft.so.7.5 locally
    I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcuda.so.1 locally
    I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcurand.so.7.5 locally
    >>>
    >>> hello = tf.constant('Hello, TensorFlow!')
    >>> sess = tf.Session()
    >>> sess.run(hello)
    Hello, TensorFlow!
    >>> a = tf.constant(10)
    >>> b = tf.constant(32)
    >>> sess.run(a+b)
    42
    >>>
```

##### Example 2: Trainning a CNN using MNIST dataset

```
    python ~/tensorflow/tensorflow/models/image/mnist/convolutional.py
```

##### Example 3:  Training a CNN the CIFAR-10 dataset

The tensorflow documentation has a great write up using this dataset:
https://www.tensorflow.org/versions/master/tutorials/deep_cnn/index.html

On a g2.2xlarge or p2.xlarge:

```
  python ~/tensorflow/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py
```

To view the GPU usage:

```
  watch -n0.1 nvidia-smi
```

On a g2.8xlarge, p2.8xlarge or p2.16xlarge:

By default the python script will use one GPU. To have the script use one or more GPUs
you need to specify the argument "--num_gpus={ Number of GPUS }" at the end of the command.


```
    python ~/tensorflow/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py --num_gpus=4
```

To view all the GPUs being used run nvidia-smi:

```
    watch -n0.1 nvidia-smi
```


TensorFlow TensorBoard
-------------------------------------------------------------------------------

Tensorboard is a suite visualization tools to help understand, debug and optimize
your program.

 * [Tensorboard Documentation](https://www.tensorflow.org/versions/r0.10/how_tos/summaries_and_tensorboard/index.html)

You can start it with the following command:

```
tensorboard --logdir <path to run log dir>

# Example (Uses the current directory)
tensorboard --logdir .
```

The service will be listening on http://{ EC2 Instance Public IP }:6006

You will need to update the security group for the EC2 instance and allow traffic
to port 6006. For more information on updating security groups please see:

 * [AWS EC2 - Adding Rules to a Security Group](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-network-security.html#adding-security-group-rule)

TensorFlow Serving
-------------------------------------------------------------------------------

Tensorflow Serving is a flexible, high-performance serving system for machine learning models.
You can read more about using it here:

 * [Github](https://github.com/tensorflow/serving)
 * [Documentation](https://tensorflow.github.io/serving/)



TensorFlow Magenta
-------------------------------------------------------------------------------

This AMI also has Tensorflow Magenta bundled with it.  A library that uses
machine learning to create compelling art and music.

 * Github: https://github.com/tensorflow/magenta
 * Blog: http://magenta.tensorflow.org/


TFLearn
-------------------------------------------------------------------------------

From the site:
```
TFlearn is a modular and transparent deep learning library built on top of Tensorflow.
It was designed to provide a higher-level API to TensorFlow in order to facilitate
and speed-up experimentations, while remaining fully transparent and compatible with it.
```

  * [Github](https://github.com/tflearn/tflearn)
  * [Documentation](http://tflearn.org/)

Keras
-------------------------------------------------------------------------------

This AMI has Keras set to use Tensorflow as it's backend.   Keras is compatible
with Python 2.7 to 3.5.

Keras is a minimalist, highly modular neural networks library, written in
Python and capable of running on top of either TensorFlow or Theano. It was
developed with a focus on enabling fast experimentation. Being able to go from
idea to result with the least possible delay is key to doing good research.

Use Keras if you need a deep learning library that:

 * allows for easy and fast prototyping (through total modularity, minimalism, and extensibility).
 * supports both convolutional networks and recurrent networks, as well as combinations of the two.
 * supports arbitrary connectivity schemes (including multi-input and multi-output training).
 * runs seamlessly on CPU and GPU.

You can read the documentation at http://keras.io/



#### Keras Tensorflow Tutorial:

Keras as a simplified interface to TensorFlow: tutorial, by Francois Chollet

  A complete guide to using Keras as part of a TensorFlow workflow:

    http://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.htmlnds-to-keras


Supported AWS Instances
-------------------------------------------------------------------------------
```
t2.nano     t2.micro    t2.medium   t2.large
m3.medium   m3.large    m3.xlarge   m3.2xlarge
m4.large    m4.xlarge   m4.2xlarge  m4.4xlarge  m4.10xlarge  m4.16xlarge
c3.large    c3.xlarge   c3.2xlarge  c3.4xlarge  c3.8xlarge
c4.large    c4.xlarge   c4.2xlarge  c4.4xlarge  c4.8xlarge
r3.large    r3.xlarge   r3.2xlarge  r3.4xlarge  r3.8xlarge
i2.xlarge   i2.2xlarge  i2.4xlarge  i2.8xlarge
d2.xlarge   d2.2xlarge  d2.4xlarge  d2.8xlarge
g2.2xlarge  g2.8xlarge
p2.xlarge   p2.8xlarge
x1.16xlarge x1.32xlarge
```

Version History
-------------------------------------------------------------------------------


v2016.08

 * Updated to Tensorflow v0.12.1
 * Added OpenCV for Python3 version 3.1.0
 * Added OpenCV for Python2 version 3.1.0


v2016.07

 * Updated to Tensorflow v0.11.0
 * Updated Matplotlib to 1.5.3
 * Added Hyperas 0.2 (Python2 and Python3 modules)
 * Added Numpy 1.11.1 (Python2 and Python3 modules)
 * Added SciPy 0.18.0 (Python2 and Python3 modules)
 * Added Pandas 0.18.1 (Python2 and Python3 modules)
 * Added Sympy 1.0 (Python2 and Python3 modules)
 * Added Enum34 (Python 2 and Python 3)
 * Added PyCuda 2016.1.2 (Python 2 and Python 3)
 * Added support to compile pycuda kernels for Jupyter
 * Added Jupyter NBExtensions
 * Jupyter start script is now called jupyter (sudo service jupyter restart)


v2016.06

 * Updated to Tensorflow 10
 * AMI now starts with 100G
 * Added support for AWS p2 instance types


v2016.05

 * Updated CuDNN to 5.1
 * Updated to ipython 5
 * Updated to latest version Bitfusion Boost 0.1.0+1561
 * Updated to sklearn 0.17.1 (Fixes udacity notebook tutorial)
 * Added Tensorflow Serving
 * Added GPU Stat
 * Added tflearn 0.2.1
 * Fixed jupyter pdf export issue


v2016.04

 * Python 3 matplotlib library added
 * Set Jupyter notebook default kernel to py2 to support udacity notebooks
 * New versioning
 * Disable warning in jupyter notebook
 * Added support for AWS p2.xlarge and p2.8xlarge


v0.03

 * Added Python 3 libraries for Tensorflow, Keras (1.0.5), Scikit-learn
 * Added Jupyter Python 3 kernel
 * Added Tensorflow Magenta
 * Updated scikit-learn


v0.02

 * Added Keras python library
 * Added Tensorflow Magenta
 * Upgrade to cudnn 5
 * Upgrade to the latest version of boost 0.1.0+1518
 * Upgrade to the latest version of Tensorflow (0.9)


v0.01

 * Nvidia Driver Version 352
 * Cuda Toolkit Version 7.5
 * Nvidia cuDNN Version 4
 * bfboost 0.1.0+1402




Support
-------------------------------------------------------------------------------

Please send all comments and support request to support@bitfusion.io

[Join us on Slack](https://slack-bitfusion-aws.herokuapp.com/)  [![](https://slack.global.ssl.fastly.net/272a/img/icons/favicon-16.png)](https://slack-bitfusion-aws.herokuapp.com/)
[Contact Us](http://www.bitfusion.io/support/)           
