# MNIST Digit Recogniser FCN with Tensorflow

![alt-text](/images/mnist_image.png)

A fully connected neural network with one layer of hidden unit with 784 weights and 10 outputs. The cross entropy loss function was used since this is a classification problem, and the Adam optimiser was used for backpropagation.

The original MNIST has a total of 70000 gray scale hand written digits image of 28 x 28 pixels. By unrolling each individual pixels, this gives us a total of 784 input features for the neural network.

The data is splitted to training (55000), test (10000) and validation sets (500). The training was done in minibatch size of 100 and a total of 100 epochs.

The final training results achieved 100% for both validation and test sets, which is of no surprise for the rather low resolution 28 x 28 image. The final cost is 0.22209.

![alt-text](/images/result_epoch100.jpg)