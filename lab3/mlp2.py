
# Modified MLP with two hidden layers
# Based on code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# Gradient derivation for the extra hidden layer:
#
# Network: Input -> Hidden1 -> Hidden2 -> Output
# Weights: weights1      weights2      weights3
#
# Forward pass:
#   hidden1 = sigmoid(inputs . weights1)
#   hidden2 = sigmoid(hidden1 . weights2)
#   output  = f(hidden2 . weights3)        where f depends on outtype
#
# Backward pass (backpropagation):
#   deltao  = dE/d(output_activation)      (same as original MLP)
#   deltah2 = hidden2 * beta * (1 - hidden2) * (deltao . weights3^T)
#   deltah1 = hidden1 * beta * (1 - hidden1) * (deltah2 . weights2^T)   <-- NEW
#
# Weight updates:
#   weights3 -= eta * (hidden2^T . deltao)  + momentum * prev_update
#   weights2 -= eta * (hidden1^T . deltah2) + momentum * prev_update
#   weights1 -= eta * (inputs^T  . deltah1) + momentum * prev_update    <-- NEW

import numpy as np

class mlp2:
    """ A Multi-Layer Perceptron with two hidden layers"""

    def __init__(self, inputs, targets, nhidden1, nhidden2, beta=1, momentum=0.9, outtype='logistic'):
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden1 = nhidden1
        self.nhidden2 = nhidden2

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        # Three sets of weights: input->hidden1, hidden1->hidden2, hidden2->output
        self.weights1 = (np.random.rand(self.nin+1, self.nhidden1)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden1+1, self.nhidden2)-0.5)*2/np.sqrt(self.nhidden1)
        self.weights3 = (np.random.rand(self.nhidden2+1, self.nout)-0.5)*2/np.sqrt(self.nhidden2)

    def earlystopping(self, inputs, targets, valid, validtargets, eta, niterations=100):
        valid = np.concatenate((valid, -np.ones((np.shape(valid)[0], 1))), axis=1)

        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000

        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001)):
            count += 1
            print(count)
            self.mlptrain(inputs, targets, eta, niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)

        print("Stopped", new_val_error, old_val_error1, old_val_error2)
        return new_val_error

    def mlptrain(self, inputs, targets, eta, niterations):
        inputs = np.concatenate((inputs, -np.ones((self.ndata, 1))), axis=1)

        updatew1 = np.zeros(np.shape(self.weights1))
        updatew2 = np.zeros(np.shape(self.weights2))
        updatew3 = np.zeros(np.shape(self.weights3))

        for n in range(niterations):
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*np.sum((self.outputs-targets)**2)
            if np.mod(n, 100) == 0:
                print("Iteration: ", n, " Error: ", error)

            # Output delta (same as original)
            if self.outtype == 'linear':
                deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
                deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata
            else:
                print("error")

            # Hidden layer 2 delta (same formula as original hidden delta)
            deltah2 = self.hidden2*self.beta*(1.0-self.hidden2)*(np.dot(deltao, np.transpose(self.weights3)))

            # Hidden layer 1 delta (NEW - backprop through weights2)
            deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltah2[:, :-1], np.transpose(self.weights2)))

            # Weight updates
            updatew1 = eta*(np.dot(np.transpose(inputs), deltah1[:, :-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden1), deltah2[:, :-1])) + self.momentum*updatew2
            updatew3 = eta*(np.dot(np.transpose(self.hidden2), deltao)) + self.momentum*updatew3
            self.weights1 -= updatew1
            self.weights2 -= updatew2
            self.weights3 -= updatew3

    def mlpfwd(self, inputs):
        # First hidden layer
        self.hidden1 = np.dot(inputs, self.weights1)
        self.hidden1 = 1.0/(1.0+np.exp(-self.beta*self.hidden1))
        self.hidden1 = np.concatenate((self.hidden1, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        # Second hidden layer
        self.hidden2 = np.dot(self.hidden1, self.weights2)
        self.hidden2 = 1.0/(1.0+np.exp(-self.beta*self.hidden2))
        self.hidden2 = np.concatenate((self.hidden2, -np.ones((np.shape(inputs)[0], 1))), axis=1)

        # Output layer
        outputs = np.dot(self.hidden2, self.weights3)

        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs), axis=1)*np.ones((1, np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print("error")

    def confmat(self, inputs, targets):
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        outputs = self.mlpfwd(inputs)

        nclasses = np.shape(targets)[1]

        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs > 0.5, 1, 0)
        else:
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0)*np.where(targets == j, 1, 0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ", np.trace(cm)/np.sum(cm)*100)
