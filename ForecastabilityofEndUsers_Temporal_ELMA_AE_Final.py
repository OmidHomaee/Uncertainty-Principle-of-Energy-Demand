import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from AE_ELM import AutoEncoder
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import math




Loads = pd.read_csv("C:/Omid2/ModiLoadsALL.csv")
DateTime=pd.to_datetime(Loads["tstp"])

#####################################

def calculate_mape(actual_values, predicted_values):
    if len(actual_values) != len(predicted_values):
        raise ValueError("Input lists must have the same length.")
    
    absolute_percentage_errors = []
    for actual, predicted in zip(actual_values, predicted_values):
        if actual != 0:  # Avoid division by zero
            absolute_percentage_errors.append(abs((actual - predicted) / actual))
        
    if len(absolute_percentage_errors) > 0:
        mape = (sum(absolute_percentage_errors) / len(absolute_percentage_errors)) * 100
        return mape
    else:
        return None  # Return None or handle the case where there are no valid errors
####################################
def calculate_mae(actual_values, predicted_values):
    if len(actual_values) != len(predicted_values):
        raise ValueError("Input lists must have the same length.")
    
    absolute_errors = [abs(actual - predicted) for actual, predicted in zip(actual_values, predicted_values)]
    mae = sum(absolute_errors) / len(actual_values)
    return mae

####################################
def Forcasting_ELMAE(load, gg ):
    
    TrPercent = 90
    NhidNeuron = 40
    #dataLMP = pd.read_excel('DataPriceDA_AllStates_Ver3.xlsx', 4)
    NIter = 1
    mseTrainK = np.zeros([NIter,1])
    mseTestK = np.zeros([NIter,1])
    # Find indices of NaN values
    nan_indices = np.where(np.isnan(load))

    # Create a new array excluding NaN values
    demand = load[~np.isnan(load)]
    demand = demand.reset_index()
    demand=demand.drop("index",axis=1)
    demand=demand.iloc[:, 0].values


    for Iter in range(NIter): 
       # price0 = dataLMP['LBMP']
       # print(price0)
        
      #  price = np.zeros(price0.shape)
        #for k in range(len(price0)):
        #    MinV = min(price0)
        #    A = (price0[k]- MinV)/(max(price0)-MinV )
        #    price[k] = A 
        
        #price = 2*price -1
     
       
        MaxStepAhead =12 #math.floor(48/gg) 
        MaxStepBefore =24 # 7*math.floor(48/gg) 
        
        Length = len(demand)
        InputData = np.zeros([Length - MaxStepBefore , MaxStepBefore])
        OutputData =np.zeros([Length - MaxStepBefore , MaxStepAhead])
        
        
        for i in range(0,Length - MaxStepBefore):
            InputData[i] = demand[i:MaxStepBefore + i]
            OutputData[i] = demand[MaxStepBefore + i ]    
            
     
            
        InputNum =  InputData.shape[1]
        #OutputNum = OutputData.shape[1] 
        DataNum = InputData.shape[0]
        TrNum = round(DataNum*TrPercent/100)
        TestNum = DataNum - TrNum
        
        P_Train = InputData[0:TrNum]
        T_Train = OutputData[0:TrNum]
        
        P_Test = InputData[TrNum:DataNum + 1]
        T_Test = OutputData[TrNum:DataNum + 1]  
        
        
        #plt.figure(1)
        #x1 = np.arange(1,TrNum+1,1)
        #x2 = np.arange(1,TestNum+1,1)
        
        #plt.subplot(2,1,1)
        #plt.plot(x1,T_Train)
       # plt.subplot(2,1,2)
        #plt.plot(x2,T_Test)
        
        #plt.show()
        
    
        w = AutoEncoder(TrNum, NhidNeuron, InputNum, P_Train, T_Train, MaxStepBefore, price, TrPercent)
    
        w1 = np.transpose(w)
    
    
        tempH0 = np.dot(P_Train, w1)
    
        #BiasMatrix = np.dot(np.ones([TrNum, InputNum]),Bias)
    
    
        #tempH = tempH0 + BiasMatrix
    
        H = np.sin(tempH0)
        pinv_H = np.linalg.pinv(H)
    
        w2 = np.dot(pinv_H, T_Train)
    
        O_Train = np.dot(H,w2)
    
    
        mseTrain = mean_squared_error(T_Train, O_Train)
    
    #%% Test
    
        tempH0_test = np.dot(P_Test, w1)
    
    #BiasMatrix_test = np.dot(np.ones([TestNum, InputNum]),Bias)
    
    
    #tempH_test = tempH0_test + BiasMatrix_test
    
        H_test = np.sin(tempH0_test)
        pinv_H_test = np.linalg.pinv(H_test)
    
    #w2 = np.dot(pinv_H_test, T_Test)
    
        O_Test = np.dot(H_test,w2)
        MAPE = calculate_mape(T_Test[-1],O_Test[-1])
        MAE = calculate_mae(T_Test[-1],O_Test[-1])
    return MAPE, MAE


######################

noofEndUser=200
MaxNumofTimeSteps=20
MAPE_Results=[]
MAE_Results=[]


G_list=[0,1,2] #list(range(0, MaxNumofTimeSteps))

for group in range (1,noofEndUser+1):  #  sorted_df["g"][31:]: # 
    print("EU No=", group)
    if group<42:
        Modified_Load_G = pd.read_csv("C:/Omid2/Modified_Load"+ str(group) +".csv")
    else:
        Modified_Load_G = pd.read_csv("C:/Omid2/Modified_Load24h"+ str(group) +".csv")

    
    Modified_Load_G["Time"]= DateTime
    Modified_Load_G=Modified_Load_G.drop("Unnamed: 0",axis=1)
    normalized_lg=pd.DataFrame()
    aa=Modified_Load_G[Modified_Load_G.columns.values[0]].first_valid_index()
    bb=Modified_Load_G[Modified_Load_G.columns.values[0]].last_valid_index()
    for ff in Modified_Load_G.columns.values[G_list]:
        a=Modified_Load_G[ff].first_valid_index()
        aa=max(aa,a)
        b=Modified_Load_G[ff].last_valid_index()
        bb= min(bb,b)
    Modified_Load_G=Modified_Load_G[aa:bb]
    normalized_lg["Time"]=Modified_Load_G["Time"]
    num = 0
    for gg in Modified_Load_G.columns.values[G_list]:
        normalized_lg[gg] = (Modified_Load_G[gg] - Modified_Load_G[gg].min()) / (Modified_Load_G[gg].max() - Modified_Load_G[gg].min()) 
        row_num = len(normalized_lg[gg])
       # print("size of dataset is =" , row_num, "time step=", 30*(G_list[num]+1))
        MAPE, MAE = Forcasting_ELMAE(normalized_lg[gg], G_list[num]+1)
        MAPE_Results.append({"MAPE": MAPE, "NumberofStepTime": G_list[num]+1, "Index": gg, "EU_Number": group, "Model": "ELMA"})
        MAE_Results.append({"MAE": MAE, "NumberofStepTime": G_list[num]+1, "Index": gg, "EU_Number": group,"Model": "ELMA"})
        num = num+1
        
#############################





MAPE_Results_DF = pd.DataFrame(MAPE_Results) 
MAE_Results_DF = pd.DataFrame(MAE_Results)  
MAPE_Results_DF["timestep"]=MAPE_Results_DF["NumberofStepTime"]*30
MAPE_Results_DF["timeres"]=1/(MAPE_Results_DF["NumberofStepTime"]*0.5)

MAPE_Results_DF.to_csv('MAPE_ELMAE_Results_Temporal.csv') 
MAE_Results_DF.to_csv('MAE_ELMAE_Results_Temporal.csv') 


