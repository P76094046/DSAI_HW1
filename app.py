# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:17:52 2021

@author: algo
"""

# C:\Users\algo\Documents\Python Scripts\AI_HW1.py

# You can write code above the if-main block.
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = len(X) - 7
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit()
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	rmse = sqrt(mean_squared_error(test, predictions))
	return rmse
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					rmse = evaluate_arima_model(dataset, order)
					if rmse < best_score:
						best_score, best_cfg = rmse, order
					print('ARIMA%s RMSE=%.3f' % (order,rmse))
				except:
					continue
	print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    
if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    
    parser.add_argument('--training2',
                       default='training_data2.csv',
                       help='input training data file name')


    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    
    import pandas as pd
    from pandas.plotting import lag_plot
    from pandas.plotting import autocorrelation_plot
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.tsa.ar_model import AutoReg
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_squared_error
    from math import sqrt
    import csv
    import numpy as np
    
    # training = 2021.csv 
    data1 = pd.read_csv(args.training, header = 0, index_col = 0, parse_dates = True, squeeze=True, encoding='utf-8')
    data1.index = data1.index.to_period('W')
    n = np.where(data1.index == '2021-02-01')[0][0]
    data1['備轉容量(MW)'] = data1['備轉容量(萬瓩)'] * 10
    data1 = data1[n:]
    
    # training2 = 2020to2021.csv
    data2 = pd.read_csv(args.training2, header = 0, index_col = 0, parse_dates = True, squeeze=True)
    data2.index = data2.index.to_period('W')
    
    data = pd.concat([data2, data1], axis=0)
    data = data['備轉容量(MW)']
    print(data)
    
    # lag_plot(data)
    # plt.show()

    # autocorrelation_plot(data)
    # plt.show()

    # plot_acf(data , lags=31)
    # plt.show()

    data = data.values
    index = len(data) - 7
    train = data[1:index]
    test = data[index:]

    #  AR model
    window = 7
    model = AutoReg(train, lags=7)
    model_fit = model.fit()
    coef = model_fit.params

    # walk forward over time steps in test
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
    	length = len(history)
    	lag = [history[i] for i in range(length-window,length)]
    	yhat = coef[0]
    	for d in range(window):
    		yhat += coef[d+1] * lag[window-d-1]
    	obs = test[t]
    	predictions.append(yhat)
    	history.append(obs)
    	print('predicted=%f, expected=%f' % (yhat, obs))
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot
    '''
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.title('AR model')
    plt.show()
    '''
    #  ARIMA model
    p_values = [0, 1, 2, 4, 6, 8, 10]
    d_values = range(0, 3)
    q_values = range(0, 3)
    # evaluate_models(data, p_values, d_values, q_values)     # Grid Search
    
    size = len(data) - 7
    # size = len(data) - 14
    train, test = data[0:size], data[size:len(data)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
    	model = ARIMA(history, order=(10,2,0))
    	model_fit = model.fit()
    	output = model_fit.forecast()
    	yhat = output[0]
    	predictions.append(yhat)
    	obs = test[t]
    	history.append(obs)
    	print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test[0:6], predictions[1:7]))
    print('Test RMSE: %.3f' % rmse)
    predictions = pd.DataFrame(predictions)
    # plot forecasts against actual outcomes
    '''
    plt.plot(test)
    plt.plot(predictions.shift(-1), color='red')
    plt.title('ARIMA model')
    plt.show()    
       ''' 
    # for forecasting 3/23 ~ 3/29
    train = data
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(9):
    	model = ARIMA(history, order=(10,2,0))
    	model_fit = model.fit()
    	output = model_fit.forecast()
    	yhat = output[0]
    	predictions.append(yhat)
    	history.append(yhat)
    	# print('predicted=%f' % yhat)
    
    predictions = predictions[2:9]
    '''
    plt.plot(predictions, color='red')
    plt.title('ARIMA model')
    plt.show()
    '''

    with open('submission.csv', 'w', newline='') as csvfile:
    
        writer = csv.writer(csvfile)
        writer.writerow(['date', 'operating_reserve(MW)'])
        for i in range(len(predictions)):
            writer.writerow(['2021032%s'%(i+3), predictions[i]])
    
    
    
    
    
    
    
    
    
