__author__ = 'alexjzy'
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout
from keras import regularizers

class NeuralNetwork():
    def __init__(self):
        self.EPOCHS = 30
        self.BATCH_SIZE = 5
        self.DROPOUT_RATIO = 0.25
        self.HIDDEN_NODES = 7
        self.OUTPUT_NODES = 1


    def construct(self, train_X, train_Y, test_X, test_Y):
        model = Sequential()
        model.add(Dense(train_X.shape[1],
                        input_dim = train_X.shape[1],
                        kernel_initializer='random_normal',
                        activation='relu'))
        model.add(Dropout(self.DROPOUT_RATIO))
        model.add(Dense(self.HIDDEN_NODES, kernel_initializer='random_normal', activation='relu'))
        model.add(Dense(self.OUTPUT_NODES, kernel_initializer='random_normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')





