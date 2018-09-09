from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.layers.convolutional_recurrent import ConvLSTM2D
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, Bidirectional, AveragePooling2D
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import Orthogonal
from keras.regularizers import l2
from keras.utils import plot_model
import keras.utils.vis_utils
from os import path
from glob import glob

np.random.seed(0)
def load_train_data():
    base_dir = path.abspath(path.dirname(__file__))
    relative_path = '806_data'
    regex_full_path = path.join(base_dir, relative_path + "/*")
    file_names = glob(regex_full_path)

    X = pd.DataFrame()
    Y = []
    DataRef = pd.read_excel(file_names[0], header=None)
    Xref = DataRef.loc[0:7499]
    for i in range(len(file_names)):
        data = pd.read_excel(file_names[i], header=None)
        X = pd.concat([X, data.loc[0:7499]])
        y = data.loc[7500, 0][9:]
        y = float(y)
        Y.append(y)
    Xscale = preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0)
    Xscale = np.resize(Xscale, (len(file_names), Xref.shape[0], Xref.shape[1], 1))
    Y = np.array(Y)
    X = Xscale
    X = X.reshape(40, 4, 4, 1875, 1)
    Y = Y.reshape(40, )
    return X,Y

def load_test_data():
    base_dir = path.abspath(path.dirname(__file__))
    relative_path = '831_data'
    regex_full_path = path.join(base_dir, relative_path + "/*")
    file_names = glob(regex_full_path)

    X = pd.DataFrame()
    DataRef = pd.read_excel(file_names[0], header=None)
    Xref = DataRef.loc[0:7499]
    for i in range(len(file_names)):
        data = pd.read_excel(file_names[i], header=None)
        X = pd.concat([X, data.loc[0:7499]])
    Xscale = preprocessing.minmax_scale(X, feature_range=(0, 1), axis=0)
    Xscale = np.resize(Xscale, (len(file_names), Xref.shape[0], Xref.shape[1], 1))
    X = Xscale
    X = X.reshape(10, 4, 4, 1875, 1)
    return X

def CNN_LSTM2D():
    model = Sequential()

    model.add(Bidirectional(ConvLSTM2D(filters=2**5, kernel_size=(3, 3),kernel_initializer='glorot_uniform',recurrent_initializer='zero',bias_initializer='zero',
                   input_shape=(None, 4, 1500, 1),padding='same', return_sequences=False,activation='tanh',kernel_regularizer=l2(0.001),bias_regularizer=l2(0.001),activity_regularizer=l2(0.001)
                        ,recurrent_regularizer=l2(0.001)  ,go_backwards=True,dilation_rate=(1, 1))))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(2 ** 7, activation='tanh'))
    model.add(Dense(2 ** 6, activation='tanh'))
    model.add(Dense(2**5,activation='tanh'))
    model.add(Dense(2 **4, activation='tanh'))
    model.add(Dense(2 ** 3, activation='tanh'))
    model.add(Dense(2 ** 2, activation='tanh'))
    model.add(Dense(2 ** 1, activation='tanh'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer=Adam(lr=0.001))
    return model

def training():
    x_train,y_train = load_train_data()
    num_epochs = 1000 # 設定訓練週期
    num_batch_size = 40  # 設定訓練大小
    model = CNN_LSTM2D()
    callback = EarlyStopping(monitor="loss", patience=100, verbose=0, mode="min")
    train_history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=num_batch_size, callbacks=[callback], verbose=1)
    model.save('125700_DMPL_model.h5') #HDF5,pip install h5py
    del model

def predicting():
    model = load_model('125700_DMPL_model.h5')
    X_test = load_test_data()
    y_predic = model.predict(X_test)
    y_predic = np.reshape(y_predic,(10,))
    print(y_predic.shape)
    predict = {'預測值': y_predic}
    df=pd.DataFrame(data=predict)
    df.to_excel('result.xlsx')
    print(df)

def main():
    # training()
    predicting()

if __name__ == '__main__':
    main()
