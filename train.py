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

import reporting


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


def train_model(X_train, y_train, epochs):
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

    regressor.fit(X_train, y_train, epochs=epochs, batch_size=32)

    return regressor


def inverse_scale_col(scaler, arr):
    """Use a scaler to invert a single column, even if scaler expects a different shape

    :param scaler: <sklearn scaler>
    :param arr:
    :return:
    """
    m_ = np.zeros((arr.shape[0], scaler.min_.shape[0]))
    m_[:, 0] = arr[:, 0]
    m_ = scaler.inverse_transform(m_)
    return m_[:, 0]


def save_plots(predictions, target_dir):
    """Generate predictions on test set and create plots"""
    for ticker in predictions:
        preds = predictions[ticker]

        fig = plt.figure(figsize=(16, 10))

        plt.plot(preds['y_pred'], color='green', label="predicted stock price")
        plt.plot(preds['y_flat'], color='lightblue', label="baseline prediction (flat) stock price")
        plt.plot(preds['y_last'], color='blue', label="baseline prediction (linear) stock  price")
        plt.plot(preds['y'], color='black', label="actual stock price")
        plt.title(f"{ticker} Stock Price Prediction")
        plt.xlabel('Time')
        plt.ylabel(f"{ticker} Stock Price")
        plt.legend()
        fig.savefig(os.path.join(target_dir, f"stock_predictions_{ticker}.png"))


def save_log(tickers, date_range, start_time, metrics, preds, epochs, target_dir):
    """Save a log about the training session"""

    # Make JSON serializable
    for k in preds:
        for e in preds[k]:
            preds[k][e] = list(preds[k][e])

    out = {
        'tickers': tickers,
        'start_date': date_range[0].strftime("%Y-%m-%d"),
        'end_date': date_range[1].strftime("%Y-%m-%d"),
        'start_time': start_time,
        'total_time': time.time() - start_time,
        'metrics': metrics,
        'predictions': preds,
        'epochs': epochs
    }

    target_fn = os.path.join(target_dir, "train.log")
    print(f"Saving log to {target_fn} ...", end="")

    with open(target_fn, 'w') as f:
        f.write(json.dumps(out))

    print("Done")


def base_predict(X_test, target_col=0):
    """Emulates a model for baseline prediction"""
    flat = []
    last = []

    for seq in X_test:
        flat.append(seq[-1][target_col])
        last.append(2 * seq[-1][target_col] - seq[-2][target_col])

    return np.expand_dims(np.array(flat), axis=1), np.expand_dims(np.array(last), axis=1)


def evaluate_model(model, X_test, y_test, ticker_data, scalers, n_days=14):
    """Evaluates the model

    Calculates the following metrics for the first n_days days for each ticker
     - err
     - MAPE

    Calculates two baselines to compare the metrics to:
     - baeline_flat: predicts the last known value
     - baseline_last: predict the last difference for each next step

    :return: metrics
    """
    metrics = {}
    preds = {}

    for ticker in ticker_data:
        # find first entry of ticker in test data
        for idx, seq in enumerate(X_test):
            m = seq[0, ticker['idx']]
            if m == 1:
                start = idx
                break
        y_pred = model.predict(X_test[start:start + n_days])
        y_pred_flat, y_pred_last = base_predict(X_test[start:start + n_days])

        sc = scalers[ticker['name']]

        y_pred = inverse_scale_col(sc, y_pred)
        y_pred_flat = inverse_scale_col(sc, y_pred_flat)
        y_pred_last = inverse_scale_col(sc, y_pred_last)
        y = inverse_scale_col(sc, y_test[start:start + n_days])

        preds[ticker['name']] = {
            'y': y,
            'y_pred': y_pred,
            'y_flat': y_pred_flat,
            'y_last': y_pred_last,
        }

        metrics[ticker['name']] = {
            'err': [float(x) for x in list(y_pred - y)],
            'MAPE': [float(x) for x in list(np.abs(y_pred - y) / y)],
            'err_base_flat': [float(x) for x in list(y_pred_flat - y)],
            'MAPE_base_flat': [float(x) for x in list(np.abs(y_pred_flat - y) / y)],
            'err_base_last': [float(x) for x in list(y_pred_last - y)],
            'MAPE_base_last': [float(x) for x in list(np.abs(y_pred_last - y) / y)],
        }

    return metrics, preds


def train(tickers, date_range, epochs):
    """The main function

    Calls functions for the following steps:
     - Create and train an LSTM model
     - Evaluate model
     - create and save plots and log

    :param tickers: <list> ticker names corresponding to dataset
    :param date_range: <tuple of len 2> holds two datetime objects, start and end date
    :param epochs: <int> number of epochs to train for
    :return:
    """
    start_time = time.time()
    df = load_stocks(tickers, date_range=date_range)

    X_train, y_train, X_test, y_test, ticker_data, scalers = prepare_data(df)

    regressor = train_model(X_train, y_train, epochs)

    metrics, preds = evaluate_model(regressor, X_test, y_test, ticker_data, scalers)

    # create target dir for output
    target_dir = os.path.join("output", str(int(time.time()*1000)))
    os.makedirs(target_dir, exist_ok=True)

    regressor.save(os.path.join(target_dir, "regressor_model"))
    save_plots(preds, target_dir)
    save_log(tickers, date_range, start_time, metrics, preds, epochs, target_dir)


def parse_date(date_str):
    return datetime.datetime.strptime(date_str, "%d-%M-%Y")


# Create the parser
parser = argparse.ArgumentParser(description='Train a LSTM network on up to 5 tickers and a selected date range')
parser.add_argument('--tickers', '-t', type=str,
                    help='tickers to train on, separated by commas')
parser.add_argument('--dates', '-d', type=str,
                    help='start and end date for training data selection, separated by a comma')
parser.add_argument('--epochs', '-e', type=int, default=60,
                    help='number of epochs to train for')

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

    train(tickers, date_range, args.epochs)
    reporting.create_html_report()
