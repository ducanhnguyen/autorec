# autorec
Autorec is used in recommender systems. This is my implementation of autorec to predict the movies which a person can be interested based on the other people.

There are two versions of autorec: item-based autorec (i-autorec) and user-based autored (u-autorec). The authors proved that i-autorec is better than u-autorec. 

Paper: http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf

### Architecture of I-AutoRec

I-AutoRec has one hidden layer. The input and the output is the same.

<img src="https://github.com/ducanhnguyen/autorec/blob/master/img/model.png" width="450">

where R_ji = (user i, movie i), j = 1..m (m is the number of users)

### The loss function of I-AutoRec

<img src="https://github.com/ducanhnguyen/autorec/blob/master/img/loss.png" width="450">

, where the construction function is as follows:

<img src="https://github.com/ducanhnguyen/autorec/blob/master/img/reconstruction.png" width="350">

r^(i): the i-th observation

h(r^(i), theta) is the construction function

theta is parameters in the neural network (i.e., kernels W and V and biases).

# Experiments

The experiment is performed on 1M dataset. I used the same configurations as the original paper: hidden units = 500, use L2-regularization, 90% train, 10% test, activation function = identity, sigmoid.

The best training loss = 0.7714

The best validation loss = 1.1685

<img src="https://github.com/ducanhnguyen/autorec/blob/master/img/iautorec.png" width="550">

