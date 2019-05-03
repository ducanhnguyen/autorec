# autorec
This is my implementation of autorec to predict movies which a person can excited based on other people.

Autorec is used in recommender systems.

Autorec is original from autoencoder.

Paper: http://users.cecs.anu.edu.au/~akmenon/papers/autorec/autorec-paper.pdf



### Architecture of I-AutoRec

I-AutoRec has one hidden layer. The input and the output is the same.


<img src="https://github.com/ducanhnguyen/autorec/blob/master/img/model.png" width="450">

R_ji = (user i, movie i), j = 1..m (m is the number of users)

### The loss function of I-AutoRec

<img src="https://github.com/ducanhnguyen/autorec/blob/master/img/loss.png" width="450">

, where the construction function is as follows:

<img src="https://github.com/ducanhnguyen/autorec/blob/master/img/reconstruction.png" width="350">

r^(i): the i-th observation

h(r^(i), theta) is the construction function

theta is parameters in the neural network (i.e., kernels W and V and biases).

# Experiments

I used the same configurations as the original paper: hidden units = 500, use L2-regularization, 90% train, 10% test.

I test on 10M movies dataset.

Here is the RMSE of the training process and validation process.

<img src="https://github.com/ducanhnguyen/autorec/blob/master/img/iautorec.png" width="550">

Disclaimer: The best loss in my experiments is about 0.9286. It takes a whole night to train the model. However, as reported, the achievec loss in the original paper is 0.867.
