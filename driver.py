import matplotlib.pyplot as plt
import time as time

from test import *
from ann import *
from data import *

np.random.seed(499)

print('Simple method Tests\n')
test_affine_forward()
test_affine_backward()
test_relu_forward()
test_relu_backward()
test_L2_loss()
test_ANN_predict()

# Initial parameters
node_num = 30
input_num = 2
num_split = 5
epoch = 500

hidden_layer_number = [[node_num], [node_num,node_num], [node_num, node_num, node_num]]
data_set_name = ['./set1.dat', './set2.dat']
learning_rates = [1e-4, 5e-4, 1e-3]

all_net = []
_loss_train = []
_loss_valid = []

begin_test = time.time()
print('\nBegin test\n')

for i, name in enumerate(data_set_name):

    # read data sets
    begin_dataset = time.time()
    print('Begin train and test for ' + data_set_name[i][2:6] + '\n')

    X_train, Y_train, X_test, Y_test = read_data(name)

    for j, hidden in enumerate(hidden_layer_number):

        # assign hidden layer numbers
        begin_layers = time.time()
        print('Begin train and test for ' + str(len(hidden)) + ' hidden layer\n')

        loss_train_mean = np.ndarray(shape=(len(learning_rates), epoch), dtype=float)
        loss_valid_mean = np.ndarray(shape=(len(learning_rates), epoch), dtype=float)
        final_loss = np.ndarray(shape=(len(learning_rates),), dtype=float)

        for k, rate in enumerate(learning_rates):

            # cross validation part for all learning rate exists in list
            begin_rate = time.time()
            print('Begin train and validate for rate ' + str(rate) + '\n')

            _loss_train = []
            _loss_valid = []

            for l in range(num_split):
                net = ANN(hidden, input_num)

                # Split the all train data into validation and train part for cross validation
                validation_data = X_train[X_train.shape[0] / num_split * l: X_train.shape[0] / num_split * (l + 1)]
                validation_label = Y_train[Y_train.shape[0] / num_split * l: Y_train.shape[0] / num_split * (l + 1)]

                train_data = np.concatenate((X_train[0: X_train.shape[0] / num_split * l],
                                             X_train[X_train.shape[0] / num_split * (l + 1): X_train.shape[0]]),
                                            axis=0)
                train_label = np.concatenate((Y_train[0: Y_train.shape[0] / num_split * l],
                                              Y_train[Y_train.shape[0] / num_split * (l + 1): Y_train.shape[0]]),
                                             axis=0)

                # calculates the train and validate value for all splits
                loss_train, loss_valid = net.train_validate(train_data, train_label, validation_data, validation_label,
                                                            epoch, rate)

                _loss_train.append(loss_train)
                _loss_valid.append(loss_valid)

            # calculates the mean off 5 splits reult
            loss_train_mean[k] = np.mean(np.array(_loss_train, dtype=float), axis=0)
            loss_valid_mean[k] = np.mean(np.array(_loss_valid, dtype=float), axis=0)
            final_loss[k] = loss_valid_mean[k][-1]
            print('final loss: ' + str(final_loss[k]) + '\n')

            end_rate = time.time()
            print('Finish train and validate for rate ' + str(rate) + ' time: ' +
                  str(end_rate - begin_rate) + ' seconds\n')

        # plots the mean value of loss and validation results for best rates
        plt.plot(np.arange(epoch), loss_train_mean[np.argmin(final_loss)], label='train')
        plt.plot(np.arange(epoch), loss_valid_mean[np.argmin(final_loss)], label='validation')
        plt.legend(loc='upper right')
        plt.title('Model Loss')
        plt.savefig('Loss_' + str(len(hidden)) + '_hidden_layer&learning_rate_' +
                    str(learning_rates[int(np.argmin(final_loss))]) + '.png')
        plt.show()

        # trains the new ann with all train data
        _rate = learning_rates[int(np.argmin(final_loss))]
        net1 = ANN(hidden, input_num)
        net1.train_validate(X_train, Y_train, X_test, Y_test, epoch, _rate)

        # predicts the test data result with this model
        pred_test = net1.predict(X_test)
        dist_test = np.abs(pred_test - Y_test)

        # prints the min, max and avg reult of distance
        print('Minimum distance: ' + str(np.min(dist_test)))
        print('Maximum distance: ' + str(np.max(dist_test)))
        print('Average distance: ' + str(np.mean(dist_test)))
        print('Relative error: ' +
              str(np.max(np.abs(Y_test - pred_test) / (np.maximum(1e-8, np.abs(Y_test) + np.abs(pred_test))))))
        print('\n')

        end_layers = time.time()
        print('Finish Train and Test for ' + str(len(hidden)) + ' hidden layer time: ' +
              str(end_layers - begin_layers) + ' seconds\n')

    end_dataset = time.time()
    print('Finish train and test for ' + data_set_name[i][2:6] + ' time: ' +
          str(end_dataset - begin_dataset) + ' seconds\n')

end_test = time.time()
print('Finish test time: ' + str(end_test - begin_test) + ' seconds\n')
