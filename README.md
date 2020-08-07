# Model_Pruning

Neural Networks have 100s of layers with some redundancy or duplication. Generally, Training of large Neural Networks requires high computation power, more training time and it's too bulky and can cause problem while deploying. So, here we are using model pruning. Pruning helps us create Neural Network smaller and faster in training. This methods helps reduce the number of connections based on the Hessian matrix of the loss function.

## Impact on Size

Pruning reduce Network size, complexity and overfitting

![](https://github.com/gargarchit/Model_Pruning/blob/master/Impact_on_size.png)

## Performance

As model became sparse, the smaller model train faster, with increase in the sparsity and the task performance will degrade.

![](https://github.com/gargarchit/Model_Pruning/blob/master/Pruning_results.png)

## Accuracy

Generally, a small drop of accuracy may occur due to sparsity of model, Also the general trend is mentioned in above example for pruning results of ResNet-18 network on CIFAR 10.

# Reference
1. [Learning both Weights and Connections for Efficient Neural Networks](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf)
2. [Learning to Prune Filters in Convolutional Neural Networks](https://arxiv.org/pdf/1801.07365.pdf)
3. Overview about inbuilt function used in model pruning code : [Link](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras)

Credit: Images are taken from the reference mentioned above.
