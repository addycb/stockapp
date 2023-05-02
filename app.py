######################################
# Skeleton: Ben Lawson <balawson@bu.edu>
# Edited by: Addison Baum <atomsk@bu.edu> and Scarlett Li (scartt@bu.edu)
######################################
# Some code adapted from
# CodeHandBook at http://codehandbook.org/python-web-application-development-using-flask-and-mysql/
# and MaxCountryMan at https://github.com/maxcountryman/flask-login/
# and Flask Offical Tutorial at  http://flask.pocoo.org/docs/0.10/patterns/fileuploads/
# see links for further understanding
###################################################
#from datetime import datetime
#import flask
from flask import Flask, request, render_template
#for image uploading		
import base64
#Template from machinelearningmastery, edited by Addison
# import required modules
import tkinter as tk
from tkinter import *
from PIL import Image
from PIL import ImageTk


# prepare data for lstm
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import os
import io
import math
import numpy 
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def stockpredict(csvfile):
    dataset=read_csv(csvfile,header=0,index_col=0)
    values = dataset.values
    # integer encode direction
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[7, 8,9,10, 11]], axis=1, inplace=True)
    #print(reframed.head())
    # split into train and test sets
    values = reframed.values
    n_train_hours = math.floor(len(values)*.67)
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    #print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # design network
    model = keras.Sequential()
    model.add(layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
    model.add(layers.Dropout(.2))
    model.add(layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(layers.Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    pyplot.figure()
    pyplot.title("Loss", y=1, loc='center')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    #filename="postgraph"+filename[len(filename)-1][:-4]+".png"
    image = io.BytesIO()
    pyplot.savefig(image, format='jpg')
    image=image.getvalue()
    lossgraph=image
   #print(postgraph)   
    # make a prediction
    yhat = model.predict(test_X)
    print(len(yhat),len(test_X),"DATASET LENGHTS")
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = layers.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = layers.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    pyplot.figure()
    pyplot.title("Prediction, rmse: "+str(rmse), y=1, loc='center')
    pyplot.plot(test_y,label="prediction")
    pyplot.plot(yhat, label='true data')
    pyplot.legend()
    image = io.BytesIO()
    pyplot.savefig(image, format='jpg')
    image=image.getvalue()
    accgraph=image
    return rmse,lossgraph,accgraph

def stockgraphs(csvfile):
    dataset=read_csv(csvfile,header=0,index_col=0)
    values=dataset.values
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    groups = [0, 1, 2, 3, 4, 5]
    pyplot.figure()
    i=1
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group], y=0.5, loc='right')
        i += 1
    image = io.BytesIO()
    pyplot.savefig(image, format='jpg')
    image=image.getvalue()
    pregraph=image
    return pregraph



app = Flask(__name__)
app.secret_key = 'super secret string'  # Change this!

@app.route("/", methods=['GET'])
def hello():
	return render_template('stockshow.html')


@app.route("/stockshow", methods=['GET'])
def noargs():
	return render_template('stockshow.html')



@app.route("/datastandards", methods=['GET'])
def datas():
	return render_template('datastandards.html')

@app.route("/stockshow", methods=['POST'])
def set_comment():
	#print(request.files)
	stockpath=request.files.get("myfile")
	filename=stockpath.filename
	stockpath.save(filename)
	pregraph=stockgraphs(filename)
	RMSE,lossgraph,accgraph=stockpredict(filename)
	#print(type(pregraph),type(lossgraph))
	#print(pregraph,lossgraph)
	return render_template('stockshow.html',pregraph=pregraph,lossgraph=lossgraph,accgraph=accgraph,rmse=0,base64=base64)
if __name__ == "__main__":
	#this is invoked when in the shell  you run
	#$ python app.py
	app.run(port=5000, debug=True)

