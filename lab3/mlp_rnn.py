
# Task 4 - Recurrent MLP
# Based on code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)
#
# Modification: outputs at time t are fed back as additional inputs at time t+1.
# The network has (nin + nout) actual inputs: the original inputs plus the previous outputs.

import numpy as np

class mlp_rnn:
    """ A Recurrent MLP — outputs at time t fed back as inputs at time t+1 """

    def __init__(self, inputs, targets, nhidden, beta=1, momentum=0.9, outtype='logistic'):
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        # weights1 now has (nin + nout + 1) rows to accommodate recurrent inputs + bias
        self.weights1 = (np.random.rand(self.nin+self.nout+1, self.nhidden)-0.5)*2/np.sqrt(self.nin+self.nout)
        self.weights2 = (np.random.rand(self.nhidden+1, self.nout)-0.5)*2/np.sqrt(self.nhidden)

    def mlpfwd_single(self, input_with_bias):
        """ Forward pass for a single sample (1 x (nin+nout+1)) """
        hidden = np.dot(input_with_bias, self.weights1)
        hidden = 1.0/(1.0+np.exp(-self.beta*hidden))
        hidden = np.concatenate((hidden, [-1.0]))

        output = np.dot(hidden, self.weights2)

        if self.outtype == 'linear':
            return output, hidden
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*output)), hidden
        else:
            return output, hidden

    def mlptrain(self, inputs, targets, eta, niterations):
        updatew1 = np.zeros(np.shape(self.weights1))
        updatew2 = np.zeros(np.shape(self.weights2))

        for n in range(niterations):
            total_error = 0.0
            # Accumulated gradients over the sequence
            gradw1 = np.zeros(np.shape(self.weights1))
            gradw2 = np.zeros(np.shape(self.weights2))

            # Initialise recurrent output to zeros
            prev_output = np.zeros(self.nout)

            for t in range(self.ndata):
                # Build input: [original_input, previous_output, bias]
                full_input = np.concatenate((inputs[t], prev_output, [-1.0]))

                # Forward pass
                output, hidden = self.mlpfwd_single(full_input)
                prev_output = output.copy()

                # Error
                error = output - targets[t]
                total_error += 0.5*np.sum(error**2)

                # Output delta
                if self.outtype == 'linear':
                    deltao = error / self.ndata
                elif self.outtype == 'logistic':
                    deltao = self.beta * error * output * (1.0 - output)
                else:
                    deltao = error / self.ndata

                # Hidden delta
                deltah = hidden * self.beta * (1.0 - hidden) * (np.dot(deltao, np.transpose(self.weights2)))

                # Accumulate gradients
                gradw1 += np.outer(full_input, deltah[:-1])
                gradw2 += np.outer(hidden, deltao)

            if np.mod(n, 100) == 0:
                print("Iteration: ", n, " Error: ", total_error)

            updatew1 = eta * gradw1 + self.momentum * updatew1
            updatew2 = eta * gradw2 + self.momentum * updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2

    def earlystopping(self, inputs, targets, valid, validtargets, eta, niterations=100):
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
            new_val_error = self.compute_error(valid, validtargets)

        print("Stopped", new_val_error, old_val_error1, old_val_error2)
        return new_val_error

    def compute_error(self, inputs, targets):
        prev_output = np.zeros(self.nout)
        total_error = 0.0
        for t in range(len(inputs)):
            full_input = np.concatenate((inputs[t], prev_output, [-1.0]))
            output, _ = self.mlpfwd_single(full_input)
            prev_output = output.copy()
            total_error += 0.5*np.sum((targets[t] - output)**2)
        return total_error

    def predict(self, inputs):
        """ Run the recurrent network on a sequence, return all outputs """
        outputs = np.zeros((len(inputs), self.nout))
        prev_output = np.zeros(self.nout)
        for t in range(len(inputs)):
            full_input = np.concatenate((inputs[t], prev_output, [-1.0]))
            output, _ = self.mlpfwd_single(full_input)
            prev_output = output.copy()
            outputs[t] = output
        return outputs
