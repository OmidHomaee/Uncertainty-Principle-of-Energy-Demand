import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error
from AE_ELM import AutoEncoder
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties



#Loads = pd.read_csv("C:/Omid/ModiLoadsALL.csv")
#DateTime=pd.to_datetime(Loads["tstp"])

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
def Forcasting_ELMAE(load):
    
    TrPercent = 90
    NhidNeuron = 200
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
       
        MaxStepAhead = 12
        MaxStepBefore = 24
        
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
        
    
        w = AutoEncoder(TrNum, NhidNeuron, InputNum, P_Train, T_Train, MaxStepBefore, demand, TrPercent)
    
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
###################################


Noofgourps=220
maxnumofcusingroups=20
MAPE_Results=[]
MAE_Results=[]


G_list= [0,  3,  7,  11,  15,  19] # list(range(0, maxnumofcusingroups))
for group in range (1,Noofgourps+1):
    print("Group No=", group)
    Modified_Load_G = pd.read_csv("C:/Omid/Forecastability of End Users_Part2_NE results/G"+ str(group) +".csv")
    #Modified_Load_G["Time"]= DateTime
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
    #normalized_lg["Time"]=Modified_Load_G["Time"]
    num = 0
    for gg in Modified_Load_G.columns.values[G_list]:
        normalized_lg[gg] = (Modified_Load_G[gg] - Modified_Load_G[gg].min()) / (Modified_Load_G[gg].max() - Modified_Load_G[gg].min()) 
        MAPE, MAE = Forcasting_ELMAE(normalized_lg[gg])
        MAPE_Results.append({"MAPE": MAPE, "NumberofEU": G_list[num], "Index": gg, "Group_Number": group, "Model": "AE_ELM"})
        MAE_Results.append({"MAE": MAE, "NumberofEU": G_list[num], "Index": gg, "Group_Number": group,"Model": "AE_ELM"})
        num = num+1
        
#############################


maxncing=20

font = FontProperties()
font.set_family('serif')
font.set_name('Arial')
font.set_size(12)
font.set_weight('bold')




MAPE_Results_DF = pd.DataFrame(MAPE_Results) 
MAE_Results_DF = pd.DataFrame(MAE_Results)   
MAPE_Results_DF.to_csv('MAPE_Results_DF_MLMAE_200_FF.csv') 
MAE_Results_DF.to_csv('MAE_Results_DF_MLMAE_200_FF.csv') 


data_Vis_groups=MAPE_Results_DF

data_Vis_groups["NumberofEU"]=data_Vis_groups["NumberofEU"]+1
data_Vis_groups["EUr"]=1/data_Vis_groups["NumberofEU"]

        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=data_Vis_groups, x="EUr",  y="MAPE", errorbar=("pi",95),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis_groups, x="EUr",  y="MAPE", errorbar=("pi",75),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis_groups, x="EUr",  y="MAPE", errorbar=("pi",50),  err_style="band", color='blue')  # Create the line plot




plt.xlabel("End user resolution",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.ylabel("MAPE",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.yticks(fontsize=12)  
#plt.xlim(1,maxncing)
plt.xlim(1/maxncing,1)
plt.xticks(fontsize=12)  


# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed

plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y 

plt.savefig('MAPE_EndUser_res_244.png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot


####################################

        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=data_Vis_groups, x="NumberofEU",  y="MAPE", errorbar=("pi",95),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis_groups, x="NumberofEU",  y="MAPE", errorbar=("pi",75),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis_groups, x="NumberofEU",  y="MAPE", errorbar=("pi",50),  err_style="band", color='blue')  # Create the line plot






plt.xlabel("Number of aggregated end users",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.ylabel("MAPE",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.yticks(fontsize=12)  
#plt.xlim(1,maxncing)
plt.xlim(1,maxncing)
plt.xticks(fontsize=12)  


# Set the x-axis tick locator to display integer values only
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed

plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y 

plt.savefig('MAPE_enduser_244.png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot
