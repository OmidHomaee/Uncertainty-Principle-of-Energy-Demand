import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsolutePercentageError


import math
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster
from tensorflow.keras.callbacks import EarlyStopping


Loads = pd.read_csv("C:/Omid/ModiLoadsALL.csv")
DateTime=pd.to_datetime(Loads["tstp"])


def calculate_mape(actual_values, predicted_values):
    if len(actual_values) != len(predicted_values):
        raise ValueError("Input lists must have the same length.")
    
    absolute_percentage_errors = []
    for actual, predicted in zip(actual_values, predicted_values):
        if actual != 0:  # Avoid division by zero
            absolute_percentage_errors.append(abs((actual - predicted) / actual))
        
    mape = (sum(absolute_percentage_errors) / len(absolute_percentage_errors)) * 100
    return mape



def calculate_mae(actual_values, predicted_values):
    if len(actual_values) != len(predicted_values):
        raise ValueError("Input lists must have the same length.")
    
    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)]
    mae = sum(absolute_errors) / len(actual_values)
    return mae



def Forcasting_LSTM(load, time):
    f1= Forecaster(y=load, current_dates=time)

    f1.set_test_length(24)       # 1. 12 observations to test the results
    f1.generate_future_dates(12) # 2. 12 future points to forecast
    f1.set_estimator('lstm')     # 3. LSTM neural network
    f1.manual_forecast(
        call_me='lstm_12lags',
        lags=24,
        epochs=10,
        validation_split=.2,
        shuffle=True,
        callbacks=EarlyStopping(
        monitor='val_loss',               
        patience=5,
        ),
        lstm_layer_sizes=(100,),  # Change the size to 100
        dropout=(0,0),
            learning_rate=0.001,  # Set the learning rate to 0.001
     #           optimizer='adam',  # Set the optimizer to Adam
                batch_size=16,  # Set the batch size to 16
                loss='mean_squared_error',  # Set the loss function to MSE
)
    n1=f1.export_fitted_vals("lstm_12lags")
    MAPE = calculate_mape(n1['Actuals'][-f1.test_length:],n1['FittedVals'][-f1.test_length:])
    MAE = calculate_mae(n1['Actuals'][-f1.test_length:],n1['FittedVals'][-f1.test_length:])
    print("MAPE=",MAPE)
    return MAPE, MAE


Noofgourps=219
maxnumofcusingroups=20
MAPE_Results=[]
MAE_Results=[]
WINDOW_SIZE = 24
Step_out=12



G_list= [0,3,7 ,11, 15, 19] # list(range(0, maxnumofcusingroups))
for group in range (1,1+Noofgourps+1):
    print("Group No=", group)
    Modified_Load_G = pd.read_csv("C:/Omid/Forecastability of End Users_Part2_NE results/G"+ str(group) +".csv")
    Modified_Load_G["Time"]= DateTime
    Modified_Load_G=Modified_Load_G.drop("Unnamed: 0",axis=1)
    normalized_lg=pd.DataFrame()
    aa=Modified_Load_G[Modified_Load_G.columns.values[0]].first_valid_index()
    bb=Modified_Load_G[Modified_Load_G.columns.values[0]].last_valid_index()
    for gg in Modified_Load_G.columns.values[G_list]:
        a=Modified_Load_G[gg].first_valid_index()
        aa=max(aa,a)
        b=Modified_Load_G[gg].last_valid_index()
        bb= min(bb,b)
    Modified_Load_G=Modified_Load_G[aa:bb]
    normalized_lg["Time"]=Modified_Load_G["Time"]
    num = 0
    datalength=bb-aa

    for gg in Modified_Load_G.columns.values[G_list]:
        normalized_lg[gg] = (Modified_Load_G[gg] - Modified_Load_G[gg].min()) / (Modified_Load_G[gg].max() - Modified_Load_G[gg].min()) 
        MAPE, MAE = Forcasting_LSTM(normalized_lg[gg], normalized_lg["Time"])
        MAPE_Results.append({"MAPE": MAPE, "NumberofEU": G_list[num], "Index": gg, "Group_Number": group, "Model": "LSTM"})
        MAE_Results.append({"MAE": MAE, "NumberofEU": G_list[num], "Index": gg, "Group_Number": group,"Model": "LSTM"})
        num = num+1
        
        
        
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm


MAPE_Results_DF = pd.DataFrame(MAPE_Results) 
MAE_Results_DF = pd.DataFrame(MAE_Results)   
MAPE_Results_DF.to_csv('MAPE_Results_DF_groups_lstm.csv') 
MAE_Results_DF.to_csv('MAE_Results_DF_groups_lstm.csv') 


sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=MAPE_Results_DF, x="NumberofEU", y="MAPE", ci="sd", err_style="band", color='blue')  # Create the line plot
# Customize font properties
font_path = fm.findfont(fm.FontProperties(family="Arial", weight="bold"))
font_prop = fm.FontProperties(fname=font_path, size=12)  # Adjust the size as needed
plt.xlabel("Number of aggregated end users",  fontproperties=font_prop)  # Add a label to the x-axis (replace with your label)
plt.ylabel("MAPE", fontproperties=font_prop)  # Add a label to the y-axis (replace with your label)
plt.xlim(1,maxnumofcusingroups)
# Set the x-axis tick locator to display integer values only
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y 
plt.savefig('MAPE_LSTM_groups_F.png',  format="png", dpi=600, bbox_inches='tight')
plt.show()  # Display the plot
        
MAPE_Results_DF["EUR"] = 1/MAPE_Results_DF["NumberofEU"]

sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=MAPE_Results_DF, x="EUR", y="MAPE", ci="sd", err_style="band", color='blue')  # Create the line plot
# Customize font properties
font_path = fm.findfont(fm.FontProperties(family="Arial", weight="bold"))
font_prop = fm.FontProperties(fname=font_path, size=12)  # Adjust the size as needed
plt.xlabel("End user resolution",  fontproperties=font_prop)  # Add a label to the x-axis (replace with your label)
plt.ylabel("MAPE", fontproperties=font_prop)  # Add a label to the y-axis (replace with your label)
plt.xlim(1/maxnumofcusingroups,1)
# Set the x-axis tick locator to display integer values only
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y 
plt.savefig('MAPE_LSTM_groups_F.png',  format="png", dpi=600, bbox_inches='tight')
plt.show()  # Display the plot
        