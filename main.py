from os import path
from glob import glob
from sklearn.preprocessing import minmax_scale
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD

def read_excels(relative_path):
    base_dir = path.abspath(path.dirname(__file__))
    regex_full_path = path.join(base_dir, relative_path + "/*")
    file_names = glob(regex_full_path)
    return file_names

def preprocess(file_names):
    X_raw = pd.DataFrame()
    Y_raw = []
    df = pd.read_excel(file_names[0], header=None) # 讀取第一個檔案作為基底
    df_first_to_end = df.loc[0:7499] # 第一行到最後一行資料 除了 「加工品質量測結果:0.xxx」 那行
 
    for i in range(len(file_names)):
        data = pd.read_excel(file_names[i], header=None)
        X_raw = pd.concat([X_raw, data.loc[0:7499]]) # 數據0~7499列是input
        y = data.loc[7500, 0][9:] # 加工品質量測結果
        Y_raw.append(y)

    X_raw.values.astype("float")
    X_scale = minmax_scale(X_raw, feature_range=(0, 1), axis=0) # (300000, 4)

    X = X_scale.reshape(40, 7500, 4, 1)
    Y = np.array(Y_raw)

    return X.astype("float"), Y.astype("float")

def main():
    # 讀取所有資料，回傳所有檔名
    relative_path="806_data"
    file_names = read_excels(relative_path)

    # 前處理
    X, Y = preprocess(file_names)

    # 切訓練集、測試集
    X_Train = X[0:35] # (35, 7500, 4, 1)
    X_Test = X[35:] # (5, 7500, 4, 1)
    Y_Train = Y[0:35] # (35, )
    Y_Train = Y_Train.reshape(35, 1) # (35, 1)
    Y_Test = Y[35:] # (5, )
    Y_Test = Y_Test.reshape(5, 1) # (5, 1)

    # 建模
    model = Sequential()
    con = Conv2D(strides=1, filters=64, kernel_size=(3,3), padding='same',
                    input_shape=(X_Train.shape[1], X_Train.shape[2],X_Train.shape[3]),
                    activation='relu')
    model.add(con)
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.summary()

    # 訓練模型
    model.compile(loss='mean_squared_error', optimizer=SGD(), metrics=['accuracy'])
    model.fit(x=X_Train, y=Y_Train, validation_split=0.2, epochs=20, batch_size=10, verbose=2)

    Y_Result = model.predict(X_Test)
    score = model.evaluate(X_Test, Y_Test, verbose=1)
    print("Y_Result: ")
    print(Y_Result)
    print("Y_Test: ")
    print(Y_Test)
    print('Test loss:', score[0])
    # print('Test accuracy', score[1])

if __name__ == '__main__':
    main()