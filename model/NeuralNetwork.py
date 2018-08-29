__author__ = 'alexjzy'
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.layers import Dropout
from keras import regularizers
from keras.models import load_model


class NeuralNetwork:
    def __init__(self, epoch, hidden, dropout, batch_size=2, output=1):
        self.EPOCHS = epoch
        self.BATCH_SIZE = batch_size
        self.DROPOUT_RATIO = dropout
        self.HIDDEN_NODES = hidden
        self.OUTPUT_NODES = output
        self._model = Sequential()

    def construct(self, train_x, train_y, test_x, test_y):
        model = Sequential()
        model.add(Dense(train_x.shape[1],
                        input_dim=train_x.shape[1],
                        kernel_initializer='random_normal',
                        activation='relu'))
        model.add(Dropout(self.DROPOUT_RATIO))
        model.add(Dense(self.HIDDEN_NODES, kernel_initializer='random_normal', activation='relu'))
        model.add(Dense(self.OUTPUT_NODES, kernel_initializer='random_normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(train_x, train_y, epochs=self.EPOCHS, validation_data=(test_x, test_y),
                  batch_size=self.BATCH_SIZE, verbose=1)
        self._model = model
        return model

    def predict(self, x):
        pred = self._model.predict(x)
        return pred

if __name__ == '__main__':
    pass
