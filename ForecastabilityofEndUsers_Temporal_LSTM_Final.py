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




def calculate_mape(actual_values, predicted_values):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).
    """
    absolute_percentage_errors = []

    for actual, predicted in zip(actual_values, predicted_values):
        if actual != 0:  # Avoid division by zero
            absolute_percentage_errors.append(abs((actual - predicted) / actual))

    if len(absolute_percentage_errors) == 0:
        print ("len(absolute_percentage_errors) == 0")
        return 0  # Handle the case where there are no non-zero actual values

    mape = (sum(absolute_percentage_errors) / len(absolute_percentage_errors)) * 100
    return mape


def calculate_mae(actual_values, predicted_values):
    if len(actual_values) != len(predicted_values):
        raise ValueError("Input lists must have the same length.")
    
    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)]
    mae = sum(absolute_errors) / len(actual_values)
    return mae

EPC=10
def Forcasting_LSTM_M(load, WINDOW_SIZE,Step_out,datalength):
           # obj="MACX1"
       # print("OBJ=",obj)
    aaa=datalength
    X1, Y1 = create_dataset(load, WINDOW_SIZE,Step_out)
    X_train11, y_train11 = X1[:int(0.8*aaa)], Y1[:int(0.8*aaa)]
    X_val11, y_val11 = X1[int(0.8*aaa):int(0.9*aaa)], Y1[int(0.8*aaa):int(0.9*aaa)]
    X_test11, y_test11 = X1[int(0.9*aaa):], Y1[int(0.9*aaa):]
    model1 = Sequential()
    model1.add(LSTM(100, activation='relu', input_shape=(WINDOW_SIZE, 1)))
    model1.add(Dense(Step_out))
    model1.compile(optimizer='adam', loss=calculate_mape)
    cp1 = ModelCheckpoint('model1/', save_best_only=True)
    model1.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])
    model1.fit(X_train11, y_train11, epochs=EPC, batch_size=16, verbose=1, validation_data=(X_val11, y_val11), callbacks=[cp1])
    model1 = load_model('model1/')
    val_predictions = model1.predict(X_val11).flatten()
    val_results = pd.DataFrame(data={'Val Predictions':val_predictions, 'Actuals':y_val11.flatten()})
    #plt.plot(val_results['Val Predictions'][:100])
    # plt.plot(val_results['Actuals'][:100])
    test_predictions = model1.predict(X_test11).flatten()
    test_results = pd.DataFrame(data={'Test Predictions':test_predictions, 'Actuals':y_test11.flatten()})
    # plt.plot(test_results['Test Predictions'][700:800])
    # plt.plot(test_results['Actuals'][700:800])
    MAPE= calculate_mape(test_results["Actuals"],test_results['Test Predictions'])
    MAE = calculate_mae(test_results["Actuals"],test_results['Test Predictions'])
    return MAPE, MAE


def create_dataset(load, window_size,step_out):   
    Datamatrix_obj_as_np =load #Datamatrix[objective].to_numpy()
    X=[]
    Y=[]
    for i in range(len(Datamatrix_obj_as_np)-(window_size+step_out)+1):
        Xrow = [[a] for a in Datamatrix_obj_as_np[i:i+window_size]]
        X.append(Xrow)
        Yrow= [[c] for c in Datamatrix_obj_as_np[i+window_size:i+window_size+step_out]]
        Y.append(Yrow)
    return np.array(X), np.array(Y)





noofEndUser=200
MaxNumofTimeSteps=48
MAPE_Results=[]
MAE_Results=[]
k=1
WINDOW_SIZE = 24
Step_out=12



G_list= [0,1,2,3,4] # list(range(0, maxnumofcusingroups))



for group in range (1,noofEndUser+1):  #  sorted_df["g"][31:]: # 
#for group in sorted_df["g"][32:]:
    print("EU No=", group)
    if group<42:
        Modified_Load_G = pd.read_csv("C:/Omid/Modified_Load"+ str(group) +".csv")
    else:
        Modified_Load_G = pd.read_csv("C:/Omid/Modified_Load24h"+ str(group) +".csv")
  #  Modified_Load_G = pd.read_csv("/kaggle/input/modifiedloads-multipla-time-steps/Modified_Load"+ str(group) +".csv")
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
    Modified_Load_G = Modified_Load_G[aa:bb]
    normalized_lg["Time"]=Modified_Load_G["Time"]
    num = 0
    datalength=bb-aa
    for gg in Modified_Load_G.columns.values[G_list]:
        normalized_lg[gg] = (Modified_Load_G[gg] - Modified_Load_G[gg].min()) / (Modified_Load_G[gg].max() - Modified_Load_G[gg].min()) 
        MAPE, MAE = Forcasting_LSTM_M(normalized_lg[gg], WINDOW_SIZE,Step_out,datalength)
        MAPE_Results.append({"MAPE": MAPE, "NumberofStepTime": G_list[num]+1, "Index": gg, "EU_Number": group, "Model": "LSTM_M"})
        MAE_Results.append({"MAE": MAE, "NumberofStepTime": G_list[num]+1, "Index": gg, "EU_Number": group,"Model": "LSTM_M"})
        num = num+1

import matplotlib.ticker as ticker
import matplotlib.font_manager as fm


MAPE_Results_DF = pd.DataFrame(MAPE_Results) 
MAE_Results_DF = pd.DataFrame(MAE_Results)   
MAPE_Results_DF.to_csv('MAPE_Results_DF_TemporalF.csv') 
MAE_Results_DF.to_csv('MAE_Results_DF_TemporalF.csv') 


sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=MAPE_Results_DF, x="NumberofStepTime", y="MAPE", ci="sd", err_style="band", color='blue')  # Create the line plot
# Customize font properties
font_path = fm.findfont(fm.FontProperties(family="Arial", weight="bold"))
font_prop = fm.FontProperties(fname=font_path, size=12)  # Adjust the size as needed
plt.xlabel("Time Step",  fontproperties=font_prop)  # Add a label to the x-axis (replace with your label)
plt.ylabel("MAPE", fontproperties=font_prop)  # Add a label to the y-axis (replace with your label)
#plt.xlim(1,MaxNumofTimeSteps)
# Set the x-axis tick locator to display integer values only
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y 
plt.savefig('MAPE_LSTM_timestep.png',  format="png", dpi=600, bbox_inches='tight')
plt.show()  # Display the plot