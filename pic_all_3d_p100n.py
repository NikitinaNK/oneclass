#import helpers21
#import helpers21
import math
from pandas import DataFrame


import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import colors
import sklearn
from sklearn import preprocessing
from sklearn.metrics import  precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import zero_one_loss
from sklearn import svm
from sklearn.datasets import make_blobs

from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from pandas import DataFrame
import numpy as np
import matplotlib.pylab as plt

import tensorflow.keras
import numpy as np
import math
import matplotlib.pylab as plt
from sklearn import svm
import numpy as np
from pandas import DataFrame
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from numba import njit
import statistics 
from mpl_toolkits.mplot3d import Axes3D



visual = True
verbose_show = False

viv = False

# generate 2d classification dataset
def datagen(x_c, y_c, n_samples, n_features):

    center = [[x_c, y_c]] if n_features == 2 else None
    X, Y = make_blobs(n_samples = n_samples, centers = center, n_features = n_features, cluster_std = 0.1)
    if n_features == 2:
        plt.figure(figsize=(12, 8))
        plt.scatter(X[:,0], X[:,1], marker='o', s=7, color = 'b', label = 'Training set')
        plt.legend(loc = 'upper left', fontsize = 12)
        plt.title('Training set')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('out/train_set.png')
        plt.show()
    np.savetxt('data.txt', X)

    return X


def ire(vector1, vector2):
    x = 0
    for i in range(len(vector1)):
        x += (vector2[i] - vector1[i])**2
    # !! Round to .xx
    ire = round(math.sqrt(x), 2)
    return ire

def ire_array(array1, array2):
    ire_list = []
    for index in range(array1.shape[0]):
        ire_list.append(ire(array1[index], array2[index]))
    ire_array = np.array(ire_list)
    return ire_array

class EarlyStoppingOnValue(tensorflow.keras.callbacks.Callback):

    def __init__(self, monitor='loss', baseline=None):
        super(tensorflow.keras.callbacks.Callback, self).__init__()
        self.baseline = baseline
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        current_value = self.get_monitor_value(logs)
        if current_value < self.baseline:
            self.model.stop_training = True

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            print(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value

#создание и обучение модели автокодировщика
def create_fit_save_ae(cl_train, ae_file, irefile, epohs, verbose_show, patience):

    size = cl_train.shape[1]
    #ans = '2'
    ans = input('Задать архитектуру автокодировщиков или использовать архитектуру по умолчанию? (1/2): ')
    if ans == '1':
        n = int(input("Задайте количество скрытых слоёв (нечетное число) : "))
        # Ниже строки читать входные данные пользователя с помощью функции map ()
        ae_arch = list(map(int, input("Задайте архитектуру скрытых слоёв автокодировщика, например, в виде 3 1 3 : ").strip().split()))[:n]
        ae = tensorflow.keras.models.Sequential()
        
        # input layer
        ae.add(tensorflow.keras.layers.Dense(size))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        
        # hidden layers
        for i in range(len(ae_arch)):
            ae.add(tensorflow.keras.layers.Dense(ae_arch[i]))
            ae.add(tensorflow.keras.layers.Activation('tanh'))
        
        # output layer
        ae.add(tensorflow.keras.layers.Dense(size))
        ae.add(tensorflow.keras.layers.Activation('linear'))
    else:
        ae = tensorflow.keras.models.Sequential()
        
        # input layer
        ae.add(tensorflow.keras.layers.Dense(size))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        
        # hidden layers
        ae.add(tensorflow.keras.layers.Dense(3))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(2))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(1))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(2))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        ae.add(tensorflow.keras.layers.Dense(3))
        ae.add(tensorflow.keras.layers.Activation('tanh'))
        
        # output layer
        ae.add(tensorflow.keras.layers.Dense(size))
        ae.add(tensorflow.keras.layers.Activation('linear'))

    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    ae.compile(loss='mean_squared_error', optimizer=optimizer)
    error_stop = 0.0001
    epo = epohs

    early_stopping_callback_on_error = EarlyStoppingOnValue(monitor='loss', baseline=error_stop)
    early_stopping_callback_on_improving = tensorflow.keras.callbacks.EarlyStopping(monitor='loss',
                                                                                           min_delta=0.0001, patience = patience,
                                                                                           verbose=1, mode='auto',
                                                                                           baseline=None,
                                                                                           restore_best_weights=False)
    history_callback = tensorflow.keras.callbacks.History()
    verbose = 1 if verbose_show else 0
    history_object = ae.fit(cl_train, cl_train,
                                batch_size= cl_train.shape[0],
                                epochs=epo,
                                callbacks=[early_stopping_callback_on_error, history_callback,
                                early_stopping_callback_on_improving],
                                verbose=verbose)
    ae_trainned = ae
    ae_pred = ae_trainned.predict(cl_train)
    ae_trainned.save(ae_file)

    IRE_array = np.round(ire_array(cl_train, ae_pred), 2)
    IREth = np.max(IRE_array)
    with open(irefile, 'w') as file:
        file.write(str(IREth))
    print()
    print()

    return ae_trainned, IRE_array, IREth

def test(y_pred, Y_test):
    y_pred[y_pred != Y_test] = -100 # find and mark classification error
    n_errors = (y_pred == -100).astype(int).sum()
    return n_errors

def predict_ae(nn, x_test, threshold):
    x_test_predicted = nn.predict(x_test)
    ire = ire_array(x_test, x_test_predicted)

    predicted_labels = (ire > threshold).astype(float)
    predicted_labels = predicted_labels.reshape((predicted_labels.shape[0], 1))
    ire = np.transpose(np.array([ire]))
    return predicted_labels, ire

def load_ae(path_to_ae_file):
    return tensorflow.keras.models.load_model(path_to_ae_file)



def square_calc(numb_square, data, ae, IRE_th, num, visual):
    # scan
    x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 1
    # print(x_min, x_max)
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    # print(y_min, y_max)
    z_min, z_max = data[:, 2].min() - 1, data[:, 2].max() + 1

    h_x = (x_max - x_min) / 50
    h_y = (y_max - y_min) / 50
    h_z = (z_max - z_min) / 50
    h_y = h_x
    h_z = h_x
    #print('ШАГ x:', h_x)
    #print('ШАГ y:', h_y)
    xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y),np.arange(z_min, z_max, h_z))
    X_plot = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

    #  получение ответов автоэнкодера
    Z, ire = predict_ae(ae, X_plot, IRE_th)
    # print('z')
    # print(Z)

    X_def = np.array([0, 0, 0], ndmin=2)
    for ind, ans in enumerate(Z):
        if ans == 0:
            # print(ans, ' kl= 1')
            # print(ind, len (svm_predicted_scan))
            X_def = np.append(X_def, [X_plot[ind]], axis=0)

    # построение областей покрытия и границ классов
    X_def = np.delete(X_def, 0, axis=0)
    Z = Z.reshape(xx.shape)

    print('X_def', X_def)
    print('shape X_def:', X_def.shape)
    print('our_data', data)
    print('shape our_data:', data.shape)


    if visual:
        fig = plt.figure(figsize=[12, 12])
        ax = Axes3D(fig)

        ax = fig.add_subplot(2,2,1,projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o', color='b', label='Training set')
        ax.scatter(X_def[:, 0], X_def[:, 1], X_def[:, 2], marker='o', c='darkorange', alpha=0.15, edgecolor='orange', linewidth=1.1, label='Classification set')

        ax.legend(loc='upper left', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax = fig.add_subplot(2, 2, 2, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o', color='b', label='Training set')
        ax.scatter(X_def[:, 0], X_def[:, 1], X_def[:, 2], marker='o', c='darkorange', alpha=0.15, edgecolor='orange', linewidth=1.1, label='Classification set')

        ax.legend(loc='upper left', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(30, -200)

        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o', color='b', label='Training set')
        ax.scatter(X_def[:, 0], X_def[:, 1], X_def[:, 2], marker='o', c='darkorange', alpha=0.15, edgecolor='orange',linewidth=1.1, label='Classification set')

        ax.legend(loc='upper left', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(100, -89)

        ax = fig.add_subplot(2, 2, 4, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], marker='o', color='b', label='Training set')
        ax.scatter(X_def[:, 0], X_def[:, 1], X_def[:, 2], marker='o', c='darkorange', alpha=0.15, edgecolor='orange',linewidth=1.1, label='Classification set')

        ax.legend(loc='upper left', fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(2, -120)
        plt.show()

        plt.plot(data[:, 1], data[:, 2], 'o', X_def[:, 1], X_def[:, 2], 'o')
        plt.show()
        plt.plot(data[:, 0], data[:, 2], 'o', X_def[:, 0], X_def[:, 2], 'o')
        plt.show()
        plt.plot(data[:, 0], data[:, 1], 'o', X_def[:, 0], X_def[:, 1], 'o')
        plt.show()



    h_x = (x_max - x_min) / numb_square
    h_y = (y_max - y_min) / numb_square
    h_z = (z_max - z_min) / numb_square

    h_x = abs(h_x)
    h_y = abs(h_y)
    h_z = abs(h_z)


    cart = np.zeros((numb_square, numb_square, numb_square))
    cart_ae = np.zeros((numb_square, numb_square, numb_square))

    xbounds = [(x_min + i * h_x, x_min + (i + 1) * h_x) for i in range(numb_square)]
    ybounds = [(y_min + i * h_y, y_min + (i + 1) * h_y) for i in range(numb_square)]
    zbounds = [(z_min + i * h_z, z_min + (i + 1) * h_z) for i in range(numb_square)]

    def index_of_dot(bounds, dot_coord):
        for index, bound in enumerate(bounds):
            if bound[0] <= dot_coord < bound[1]:
                return index

    for dot in data:
        x_index = index_of_dot(xbounds, dot[0])
        y_index = index_of_dot(ybounds, dot[1])
        z_index = index_of_dot(zbounds, dot[2])
        cart[x_index, y_index, z_index] = 1

    for dot in X_def:
        x_index = index_of_dot(xbounds, dot[0])
        y_index = index_of_dot(ybounds, dot[1])
        z_index = index_of_dot(zbounds, dot[2])
        cart_ae[x_index, y_index, z_index] = 1

    amount = np.count_nonzero(cart)
    amount_ae = np.count_nonzero(cart_ae)

    # print('cart', cart)
    # print('amount: ', amount)

    # print('cart_ae', cart_ae)
    print('amount: ', amount)
    print('amount_ae: ', amount_ae)

    square_ov = amount * h_x * h_y * h_z
    square_ae = amount_ae * h_x * h_y * h_z

    print()
    print('Оценка качества AE' + str(num))
    extra_pre_ae = square_ov / square_ae
    # print('square_ov:',  square_ov)
    # print('square_ae:', square_ae)

    Ex = cart_ae - cart
    Excess = np.sum(Ex == 1) / amount
    print('IDEAL = 0. Excess: ', Excess)
    Def = cart - cart_ae
    Deficit = np.sum(Def == 1) / amount
    print('IDEAL = 0. Deficit: ', Deficit)
    cart[cart > 0] = 5
    Coa = cart - cart_ae
    Coating = np.sum(Coa == 4) / amount
    print('IDEAL = 1. Coating: ', Coating)
    summa = Deficit + Coating
    print('summa: ', summa)
    print('IDEAL = 1. Extrapolation precision (Approx): ', extra_pre_ae)
    print()
    print()


    square_ov = amount * h_x * h_y * h_z
    square_ae = amount_ae * h_x * h_y * h_z



    with open('out/result.txt', 'w') as file:
        file.write(
            '------------Оценка качества AE' + str(num) + ' С ПОМОЩЬЮ НОВЫХ МЕТРИК------------' + '\n' + \
            'Approx = ' + str(extra_pre_ae) + '\n' + \
            'Excess = ' + str(Excess) + '\n' + \
            'Deficit = ' + str(Deficit) + '\n' + \
            'Coating = ' + str(Coating) + '\n')

    return xx, yy, zz, Z

#####2D
def plot_xdef(X_train, xx, yy, Z):

    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.5)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=7, color='b')
    plt.legend(['C1'])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot2in1(X_train, xx, yy, Z1, Z2):

    plt.subplot(1, 2, 1)
    plot_xdef(X_train, xx, yy, Z1)
    plt.title('Autoencoder AE1')#. Training set. Class boundary')

    plt.subplot(1, 2, 2)
    plot_xdef(X_train, xx, yy, Z2)
    plt.title('Autoencoder AE2')#. Training set. Class boundary')
    plt.savefig('out/AE1_AE2_train_def.png')
    plt.show()


def anomaly_detection_ae(predicted_labels, ire, ire_th):
    ire = np.round(ire,2)
    ire_th = np.round(ire_th, 2)
    if predicted_labels.sum() == 0:
        print("Аномалий не обнаружено")
    else:
        print()
        print('%-10s%-10s%-10s%-10s' % ('i', 'Labels', 'IRE', 'IREth'))
        for i, pred in enumerate(predicted_labels):
            print('%-10s%-10s%-10s%-10s' % (i, pred, ire[i], ire_th))
        print('Обнаружено ', predicted_labels.sum(), ' аномалий')


def plot2in1_anomaly(X_train, xx, yy, Z1, Z2, anomalies):

    plt.subplot(1, 2, 1)
    plot_xdef(X_train, xx, yy, Z1)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], marker='o', s=12, color='r')
    plt.title('Autoencoder AE1')#. Training set. Class boundary')

    plt.subplot(1, 2, 2)
    plot_xdef(X_train, xx, yy, Z2)
    plt.scatter(anomalies[:, 0], anomalies[:, 1], marker='o', s=12, color='r')
    plt.title('Autoencoder AE2')#. Training set. Class boundary')
    plt.savefig('out/AE1_AE2_train_def_anomalies.png')
    plt.show()

def ire_plot(title, IRE_test, IREth, ae_name):

    x = range(1, len(IRE_test) + 1)
    IREth_array = [IREth for x in x]
    plt.figure(figsize = (16, 8))
    plt.title('IRE for ' + title + ' set. ' + ae_name, fontsize = 24)
    plt.plot(x, IRE_test, linestyle = '-', color = 'r', lw = 2, label = 'IRE')
    plt.plot(x, IREth_array, linestyle = '-', color = 'k', lw = 2, label = 'IREth')
    #plt.xlim(0, len(x))
    ymax = 1.5 * max(np.amax(IRE_test), IREth)
    plt.ylim(0, ymax)
    plt.xlabel('Vector number', fontsize = 20)
    plt.ylabel('IRE', fontsize = 20)
    plt.grid()
    plt.legend(loc = 'upper left', fontsize = 16)
    plt.gcf().savefig('out/IRE_' + title + ae_name + '.png')
    plt.show()

    return