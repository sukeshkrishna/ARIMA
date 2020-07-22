from flask_socketio import SocketIO, send, join_room
from flask import Flask, flash, redirect, render_template, request, session, abort,url_for
import os
#import StockPrice as SP
import re
import sqlite3
from nsepy import get_history
import matplotlib.pyplot as plt
from mpl_finance import candlestick_ohlc
import time
from datetime import date
import datetime
import pandas as pd
import numpy as np
import requests
import numpy as np
import pandas as pd
import bokeh
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.embed import components
bv = bokeh.__version__
from math import pi
from bokeh.models import HoverTool, ColumnDataSource, Label
import arima as arima

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

@app.route('/')
#loading login page or main chat page
def index():
	if not session.get('logged_in'):
		return render_template("login.html")
	else:
		return render_template('Stock.html')

@app.route('/about')
def about_page():
    return render_template("about.html")

@app.route('/registerpage',methods=['POST'])
def reg_page():
    return render_template("register.html")
	
@app.route('/loginpage',methods=['POST'])
def log_page():
    return render_template("login.html")
    
@app.route('/Back')
def back():
    return render_template("Stock.html")
    
    
@app.route('/Stock',methods=['POST'])
def Stock_page():
    	stockname=request.form['stockname']
    	startdate=request.form['startdate']
    	enddate=request.form['enddate']
    	stockname=stockname.upper()
    	print(stockname)
    	print(startdate)
    	print(enddate)
    	s=startdate.split("-")
    	e=enddate.split("-")
    	print(s[0])
    	print(s[1])
    	print(s[2])
    	
    	df = get_history(symbol=stockname,start=date(int(s[0]),int(s[1]),int(s[2])),end=date(int(e[0]),int(e[1]),int(e[2])))
    	filename=stockname+".csv"
    	df.to_csv(filename)
    	df.dropna(inplace = True)
    	
    	plot_array = np.zeros([len(df), 5])
    	plot_array[:, 0] = np.arange(plot_array.shape[0])
    	
    	print("dataset")
    	print(df)
    	print(df.iloc[:, 0:5])
    	df = pd.read_csv(filename) 
    	print(df.head())
    	df=df.drop(['Symbol','Series','Prev Close','Last','VWAP','Volume','Turnover','Trades','Deliverable Volume','%Deliverble'], axis=1)
    	df.Date = pd.to_datetime(df.Date)
    	
    	print(df.Close)
    	print(df.Date)
    	inc = df.Close > df.Open
    	dec = df.Open > df.Close
    	w = 12*60*60*1000 # half day in ms
    	
    	p = figure(plot_width=900, plot_height=500, title=stockname+"Chart", x_axis_type="datetime",x_axis_label="Days",y_axis_label="Stock Status")
    	p.line(df.Date, df.Close, line_width=2, line_color="#FB8072",legend='Price')
    	p.xaxis.major_label_orientation = pi/4
    	p.grid.grid_line_alpha=0.3
    	p.segment(df.Date, df.High, df.Date, df.Low, color="black")
    	p.vbar(df.Date[inc], w, df.Open[inc], df.Close[inc], fill_color="green", line_color="black")
    	p.vbar(df.Date[dec], w, df.Open[dec], df.Close[dec], fill_color="red", line_color="black")

    	script, div = components(p)
    	arima.process(filename)
    	return render_template('graph.html',  bv=bv,script=script, div=div)

   
@app.route('/register',methods=['POST'])
def reg():
	name=request.form['name']
	username=request.form['username']
	password=request.form['password']
	email=request.form['emailid']
	mobile=request.form['mobile']
	conn= sqlite3.connect("Database")
	cmd="SELECT * FROM login WHERE username='"+username+"'"
	print(cmd)
	cursor=conn.execute(cmd)
	isRecordExist=0
	for row in cursor:
		isRecordExist=1
	if(isRecordExist==1):
	        print("Username Already Exists")
	        return render_template("usernameexist.html")
	else:
		print("insert")
		cmd="INSERT INTO login Values('"+str(name)+"','"+str(username)+"','"+str(password)+"','"+str(email)+"','"+str(mobile)+"')"
		print(cmd)
		print("Inserted Successfully")
		conn.execute(cmd)
		conn.commit()
		conn.close() 
		return render_template("inserted.html")

@app.route('/login',methods=['POST'])
def log_in():
	#complete login if name is not an empty string or doesnt corss with any names currently used across sessions
	if request.form['username'] != None and request.form['username'] != "" and request.form['password'] != None and request.form['password'] != "":
		username=request.form['username']
		password=request.form['password']
		conn= sqlite3.connect("Database")
		cmd="SELECT username,password FROM login WHERE username='"+username+"' and password='"+password+"'"
		print(cmd)
		cursor=conn.execute(cmd)
		isRecordExist=0
		for row in cursor:
			isRecordExist=1
		if(isRecordExist==1):
			session['logged_in'] = True
			# cross check names and see if name exists in current session
			session['username'] = request.form['username']
			return redirect(url_for('index'))

	return redirect(url_for('index'))
	
@app.route("/logout")
def log_out():
    session.clear()
    return redirect(url_for('index'))

# /////////socket io config ///////////////
#when message is recieved from the client    
@socketio.on('message')
def handleMessage(msg):
    print("Message recieved: " + msg)
 
# socket-io error handling
@socketio.on_error()        # Handles the default namespace
def error_handler(e):
    pass


  
  
if __name__ == '__main__':
    socketio.run(app,debug=True,host='127.0.0.1', port=4000)
