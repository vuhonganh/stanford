#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # forward propagation
    z1 = np.dot(data, W1) + b1
    h = sigmoid(z1)
    z2 = np.dot(h, W2) + b2
    y_hat = softmax(z2)

    # get number of data in this batch
    M = y_hat.shape[0]
    cross_entropy = -np.log(y_hat)
    idx_true = np.argmax(labels, axis=1)  # index of true label is equal to index of max val
    cost = np.sum(cross_entropy[range(0, M), idx_true])
    # backward propagation
    dz2 = y_hat - labels
    gradW2 = np.dot(h.T, dz2)
    gradb2 = np.sum(dz2, axis=0)
    dh = np.dot(dz2, W2.T)
    sigm_z1 = sigmoid(z1)
    dz1 = dh * sigm_z1 * (1.0 - sigm_z1)  # element wise operators here
    gradW1 = np.dot(data.T, dz1)
    gradb1 = np.sum(dz1, axis=0)
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "My implementation shoud be fine!"


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
