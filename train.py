# Created by Sijmen van der Willik
# 07/10/2020 11:19
#
# Github: https://github.com/sijmenw/udacity-data-science-capstone

import os
import datetime
import argparse
import time
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import matplotlib.pyplot as plt


def load_stocks(tickers, date_range=None):
    """loads data for multiple tickers and combines them into a DataFrame
    date range is inclusive
    """
    data = []
    for ticker in tickers:
        df = pd.read_csv(f"./data/stocks/{ticker}.txt")
        df['ticker'] = ticker
        data.append(df)

    df = pd.concat(data)

    df['Date'] = pd.to_datetime(df['Date'])

    if date_range:
        df = df[(df['Date'] >= date_range[0]) & (df['Date'] <= date_range[1])]

    return df.drop(['OpenInt'], axis=1)


def prepare_data(df, test_fraction=0.2):
    """Prepares the data for training and testing

    Inputs are:
        'Open', 'High', 'Low', 'Close', 'Volume'
        + one additional column for each ticker (one-hot)

    Target is Open column
    """
    # one hot encoding for tickers
    # TODO check if add skip_first helps
    tickers = df['ticker'].unique()
    df = pd.get_dummies(df, columns=['ticker'])
    tick_cols = [x for x in df.columns if 'ticker_' in x]
    input_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    target_cols = ['Open']
    target_cols_idx = [input_cols.index(x) for x in target_cols]
    ticker_data = [
        {'idx': idx + len(input_cols), 'name': x.replace("ticker_", ""), 'column_name': x} for idx, x in
        enumerate(tick_cols)
    ]

    scalers = {}

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    for ticker in tickers:

        sub = df[df["ticker_" + ticker] == 1]

        train_split = int(sub.shape[0] * (1 - test_fraction))

        # min max scale per ticker
        scaler = MinMaxScaler()
        scalers[ticker] = scaler
        data = np.hstack((scaler.fit_transform(sub[input_cols]), sub[tick_cols]))

        # create sets of X and y
        for i in range(60, data.shape[0]):
            if i < train_split:
                X_train.append(data[i - 60:i])
                y_train.append(data[i, target_cols_idx])
            else:
                X_test.append(data[i - 60:i])
                y_test.append(data[i, target_cols_idx])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(input_cols + tick_cols)))

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(input_cols + tick_cols)))  # expand dims

    return X_train, y_train, X_test, y_test, ticker_data, scalers


def train_model(X_train, y_train):
    """Creates and trains LSTM regressor model"""
    regressor = Sequential()

    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    regressor.add(Dense(units=1))

    regressor.compile(optimizer='adam', loss='mean_squared_error')

    regressor.fit(X_train, y_train, epochs=60, batch_size=32)

    return regressor


def inverse_scale_col(scaler, arr):
    m_ = np.zeros((arr.shape[0], scaler.min_.shape[0]))
    m_[:, 0] = arr[:, 0]
    m_ = scaler.inverse_transform(m_)
    return m_[:, 0]


def save_plots(model, X_test, y_test, ticker_data, scalers, target_dir, n_days=14):
    """Generate predictions on test set and create plots"""
    for ticker in ticker_data:
        # find first entry of ticker in test data
        for idx, seq in enumerate(X_test):
            m = seq[0, ticker['idx']]
            if m == 1:
                start = idx
                break
        y_pred = model.predict(X_test[start:start + n_days])

        fig = plt.figure(figsize=(16, 10))
        sc = scalers[ticker['name']]
        plt.plot(inverse_scale_col(sc, y_pred), color='green',
                 label="predicted stock price")
        plt.plot(inverse_scale_col(sc, y_test[start:start + n_days]), color='black',
                 label="actual stock price")
        plt.title(f"{ticker['name']} Stock Price Prediction")
        plt.xlabel('Time')
        plt.ylabel(f"{ticker['name']} Stock Price")
        plt.legend()
        fig.savefig(os.path.join(target_dir, f"stock_predictions_{ticker['name']}.png"))


def save_log(tickers, date_range, start_time, metrics, target_dir):
    out = {
        'tickers': tickers,
        'start_date': date_range[0].strftime("%Y-%m-%d"),
        'end_date': date_range[1].strftime("%Y-%m-%d"),
        'start_time': start_time,
        'total_time': time.time() - start_time,
        'metrics': metrics
    }

    with open(os.path.join(target_dir, "train.log"), 'w') as f:
        f.write(json.dumps(out))


def evaluate_model(model, X_test, y_test, ticker_data, scalers, n_days=14):
    """Evaluates the model

    Calculates the following metrics for the first n_days days for each ticker
     - err
     - MAPE

    :return: metrics
    """
    metrics = {}

    for ticker in ticker_data:
        # find first entry of ticker in test data
        for idx, seq in enumerate(X_test):
            m = seq[0, ticker['idx']]
            if m == 1:
                start = idx
                break
        y_pred = model.predict(X_test[start:start + n_days])

        sc = scalers[ticker['name']]
        y_pred = inverse_scale_col(sc, y_pred)
        y = inverse_scale_col(sc, y_test[start:start + n_days])

        metrics['ticker'] = {
            'err': [float(x) for x in list(y_pred - y)],
            'MAPE': [float(x) for x in list(np.abs(y_pred - y) / y)]
        }

    return metrics


def train(tickers, date_range):
    start_time = time.time()
    df = load_stocks(tickers, date_range=date_range)

    X_train, y_train, X_test, y_test, ticker_data, scalers = prepare_data(df)

    regressor = train_model(X_train, y_train)

    metrics = evaluate_model(regressor, X_test, y_test, ticker_data, scalers)

    # create target dir for output
    target_dir = os.path.join("output", str(int(time.time()*1000)))
    os.makedirs(target_dir, exist_ok=True)

    regressor.save(os.path.join(target_dir, "regressor_model"))
    save_plots(regressor, X_test, y_test, ticker_data, scalers, target_dir)
    save_log(tickers, date_range, start_time, metrics, target_dir)


def parse_date(date_str):
    return datetime.datetime.strptime(date_str, "%d-%M-%Y")


# Create the parser
parser = argparse.ArgumentParser(description='Train a LSTM network on up to 5 tickers and a selected date range')
parser.add_argument('--tickers', '-t', type=str,
                    help='tickers to train on, separated by commas')
parser.add_argument('--dates', '-d', type=str,
                    help='start and end date for training data selection, separated by a comma')


if __name__ == "__main__":
    # Execute the parse_args() method
    args = parser.parse_args()
    print("Got args:", args)

    if args.tickers is None:
        tickers = ['googl.us', 'aapl.us']
        print("Using default tickers:", tickers)
    else:
        tickers = args.tickers.split(",")
        print("Using tickers:", tickers)

    if args.dates is None:
        start_date = '01-01-2015'
        end_date = '01-07-2017'
        date_range = parse_date(start_date), parse_date(end_date)
        print("Using default date range:", date_range)
    else:
        start_date, end_date = args.dates.split(",")
        date_range = parse_date(start_date), parse_date(end_date)
        print("Using date range:", date_range)

    train(tickers, date_range)
