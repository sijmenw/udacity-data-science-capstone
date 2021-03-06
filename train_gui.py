# Created by Sijmen van der Willik
# 07/10/2020 14:05
#
# Github: https://github.com/sijmenw/udacity-data-science-capstone

import os
import subprocess

import PySimpleGUI as sg

sg.theme('DarkAmber')


def get_stock_list():
    return sorted([x.replace(".txt", "") for x in os.listdir("./data/stocks")])


ticker_names = get_stock_list()

layout = [[sg.Text('Select stocks (up to 5)')],
          [sg.Text('Select stock: '), sg.Combo(ticker_names, default_value="googl.us", key='stock1')],
          [sg.Text('Select stock: '), sg.Combo(ticker_names, default_value="aapl.us", key='stock2')],
          [sg.Text('Select stock: '), sg.Combo(ticker_names, key='stock3')],
          [sg.Text('Select stock: '), sg.Combo(ticker_names, key='stock4')],
          [sg.Text('Select stock: '), sg.Combo(ticker_names, key='stock5')],
          [sg.Text('_'*30)],
          [sg.Text('Training start date (dd-mm-yyyy): '), sg.InputText(default_text="01-01-2015", key='start_date')],
          [sg.Text('Training end date (dd-mm-yyyy): '), sg.InputText(default_text="01-07-2017", key='end_date')],
          [sg.Text('_'*30)],
          [sg.Text('Dropout ratio: '), sg.InputText(default_text="0.2", key='dropout_ratio')],
          [sg.Text('Num LSTM layers: '), sg.InputText(default_text="4", key='num_lstm')],
          [sg.Text('_'*30)],
          [sg.Text('Number of training epochs: '), sg.InputText(default_text="60", key='epochs')],
          [sg.Text('_'*30)],
          [sg.Button('Train!'), sg.Button('Exit')]]

# create the Window
window = sg.Window('Stock training picker', layout)

# event Loop to process events and get values
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':	 # if user closes window or clicks cancel
        break

    print('Selected tickers:')
    tickers = [values[f'stock{i}'] for i in range(1, 6) if values[f'stock{i}'] != ""]
    print(tickers)
    print(f"{values['start_date']} -> {values['end_date']}")

    if event == "Train!":
        print("COMMAND LIST\n", ["python", "train.py",
                          "--tickers", ",".join(tickers),
                          "--dates", f"{values['start_date']},{values['end_date']}",
                          "--epochs", str(values['epochs']),
                          "--lstm", str(values['epochs']),
                          "--dropout", str(values['dropout_ratio'])])
        subprocess.Popen(["python", "train.py",
                          "--tickers", ",".join(tickers),
                          "--dates", f"{values['start_date']},{values['end_date']}",
                          "--epochs", str(values['epochs']),
                          "--lstm", str(values['num_lstm']),
                          "--dropout", str(values['dropout_ratio'])])
        window.close()

window.close()
