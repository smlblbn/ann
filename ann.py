from layers import *


class ANN(object):
    """
    This class implements a modular fully-connected neural network with an 
    arbitrary number of hidden layers, ReLU nonlinearities, and a L2 loss
    function. For a network with L layers, the architecture will be

    {affine - relu} x (L - 1) - affine - loss

    where the {...} block is repeated L - 1 times.
    """

    def __init__(self, hidden_dims, input_dim, weight_scale=1e-2, dtype=np.float32):

        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        """
        In this part of the code, all parameters of the network should be
        initialized and stored in the self.params dictionary. For a layer "l", you
        should store the weights as "Wl" and biases as "bl". For example, for the 
        first layer, "W0" and "b0" should be initialized and stored. These
        parameters will be used in training and prediction phases. Weights should
        be initialized with random numbers frome a normal distribution having zero
        mean and standard deviation equal to weight_scale argument. Biases should
        be initialized to zero.

        Inputs:
        hidden_dims: dimensions of hidden layers, a list object.
        input_dim: dimension of data vectors
        weight_scale: scale of initial random weights
        dtype: type of the parameters
        """

        ############################################################################
        #                            START OF YOUR CODE                            #
        ############################################################################
        qwe = [input_dim] + hidden_dims + [1]
        for i, neuron in enumerate(qwe[:-1]):
            self.params['W' + str(i)] = np.random.normal(0, weight_scale, (qwe[i], qwe[i + 1]))
            self.params['b' + str(i)] = np.zeros(qwe[i + 1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        This function is used for computing the loss and gradient for the network.

        Inputs:
        X: input data, an array of shape (N, d)
        y: labels, an array of shape (N,). y[i] is the label of datum X[i].

        Outputs:
        If y is None, then run a test-time forward pass of the model and return:
        - predictions: an array of shape (N,) giving target predictions, where
          preds[i] is the regression output for X[i].
        Else return:
        - loss: data loss computed using L2_loss
        - grads: a dictionary of gradients where the keys are the parameters of
          the network.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        preds = None

        """
        TODO: Implement the forward pass for the network using forward functions
        implemented in layers.py. Compute the predictions of the target variable
        for each datum and store them in the preds variable. Hint: You can use a
        dictionary like self.params to store intermediate results.
        """
        ############################################################################
        #                            START OF YOUR CODE                            #
        ############################################################################
        if self.num_layers == 1:
            preds, self.params['cache_a0'] = affine_forward(X, self.params['W0'], self.params['b0'])
        else:
            out, self.params['cache_a0'] = affine_forward(X, self.params['W0'], self.params['b0'])
            out, self.params['cache_r0'] = relu_forward(out)
            for l in range(self.num_layers - 2):
                out, self.params['cache_a' + str(l + 1)] = affine_forward(out, self.params['W' + str(l + 1)],
                                                                          self.params['b' + str(l + 1)])
                out, self.params['cache_r' + str(l + 1)] = relu_forward(out)

            preds, self.params['cache_a' + str(self.num_layers - 1)] = affine_forward(out, self.params[
                'W' + str(self.num_layers - 1)], self.params['b' + str(self.num_layers - 1)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return preds

        loss, grads = 0.0, {}

        """
        TODO: Implement the backward pass for the network using backward functions
        implemented in layers.py. Compute data loss using L2_loss and gradients
        for the parameters. Store the gradient for parameter self.params[p] in
        grads[p].
        """
        ############################################################################
        #                            START OF YOUR CODE                            #
        ############################################################################
        loss, dx = L2_loss(preds, np.reshape(y, (-1, 1)))

        if self.num_layers == 1:
            grads['dx_a0'], grads['dw_a0'], grads['db_a0'] = affine_backward(dx, self.params['cache_a0'])
        else:
            grads['dx_a' + str(self.num_layers - 1)], grads['dw_a' + str(self.num_layers - 1)], grads[
                'db_a' + str(self.num_layers - 1)] = affine_backward(dx,
                                                                     self.params['cache_a' + str(self.num_layers - 1)])
            for l in reversed(range(0, self.num_layers - 1)):
                dx = relu_backward(grads['dx_a' + str(l + 1)], self.params['cache_r' + str(l)])
                grads['dx_a' + str(l)], grads['dw_a' + str(l)], grads['db_a' + str(l)] = affine_backward(dx,
                                                                                                         self.params[
                                                                                                             'cache_a' + str(
                                                                                                                 l)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def train_validate(self, X_t, y_t, X_v, y_v, maxEpochs=100, learning_rate=1e-4):
        """
        Train the network using gradient descent algorithm.

        Inputs:
        X_t: training data, an array of shape (N, d)
        y_t: training labels, an array of shape (N,). y_t[i] is the label of datum X_t[i].
        X_v: validation data, an array of shape (M, d)
        y_v: validation labels, an array of shape (M,). y_v[i] is the label of datum X_v[i].
        maxEpochs: maximum number of epochs that will be spent for training if not
                   stopped early.
        learning_rate: hyperparameter used in traditional gradient descent algorithm

        Outputs:
        loss_train: Loss history for training set containing loss for each epoch
        loss_valid: Loss history for validation set containing loss for each epoch
        """
        loss_train = []
        loss_valid = []
        ############################################################################
        #                            START OF YOUR CODE                            #
        ############################################################################
        for i in range(maxEpochs):
            loss_t, grads = self.loss(X_t, y_t)
            loss_train.append(loss_t)

            loss_v, grads = self.loss(X_v, y_v)
            loss_valid.append(loss_v)

            for k, v in grads.iteritems():
                if k[3] == 'a':
                    if k[1] == 'w':
                        self.params['W' + str(k[4:])] = self.params['W' + str(k[4:])] - learning_rate * v
                    if k[1] == 'b':
                        self.params['b' + str(k[4:])] = self.params['b' + str(k[4:])] - learning_rate * v
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return loss_train, loss_valid

    def predict(self, X):
        """
        Predict the target variable using the trained network.

        Inputs:
        X: test data, an array of shape (N, d)

        Outputs:
        preds: an array of shape (N,) giving target predictions, where
               preds[i] is the regression output for X[i].
        """
        return self.loss(X, None)
