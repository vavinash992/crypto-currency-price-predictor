from locale import currency
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
from flask import Flask,redirect, request,url_for,render_template
app=Flask(__name__)

def normalise_zero_base(df):
    return df / df.iloc[0] - 1

def extract_window_data(df, window_len=5, zero_base=True):
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(df,train,test,window_len=10, zero_base=True, test_size=0.2):
    X_train = extract_window_data(train, window_len, zero_base)
    X_test = extract_window_data(test, window_len, zero_base)
    y_train = train["Close"][window_len:].values
    y_test = test["Close"][window_len:].values
    if zero_base:
        y_train = y_train / train["Close"][:-window_len].values - 1
        y_test = y_test / test["Close"][:-window_len].values - 1

    return X_train, X_test, y_train, y_test

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/model/<string:date>/<string:crypto>")
def model(date,crypto):
    print(date,crypto)
    crypto_currency=crypto
    against_currency="USD"
    year,month,day = date.split('-')
    start=dt.datetime((int(year)-1),int(month),int(day))
    if int(month)<12:
        end=dt.datetime(int(year),(int(month)+1),int(day))
    else:
        end=dt.datetime(int(year)+1,1,int(day))
    hist=web.DataReader(f'{crypto_currency}-{against_currency}', 'yahoo', start, end)
    hist=hist.drop(["Open","High","Low","Adj Close","Volume"],axis=1)
    test_size = 0.2
    split_row = len(hist) - int(test_size * len(hist))
    train = hist.iloc[:split_row]
    test = hist.iloc[split_row:]
    window_len=5
    zero_base=True
    X_train,X_test,Y_train,Y_test = prepare_data(hist,train,test,window_len=window_len, zero_base=zero_base, test_size=test_size)
    model = Sequential()
    model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    model.compile(loss="mse", optimizer="adam")
    model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1, shuffle=True)
    targets = test["Close"][window_len:]
    preds = model.predict(X_test).squeeze()
    preds = test["Close"].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    k=year+"-"+month+"-"+day
    preds[k]
    
    return render_template("model.html",crypto=crypto_currency,date=date,predicted=preds[k])

@app.route("/action",methods=["POST","GET"])
def submit():
    if request.method=="POST":
        date=request.form["Date"]
        crypto,currency= (request.form["crypto"]).split("-")
    return redirect(url_for("model",date=date,crypto=crypto))

if __name__ == "__main__":
    app.run(debug=True)