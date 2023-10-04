
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import math
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties




Sorted_Cust = pd.read_csv("C:/Omid/Sorted_Cust.csv")
Loads = pd.read_csv("C:/Omid/ModiLoadsALL.csv")



mtstpMax=48
def CoVfunctin(LoadV):
    Modified_Load=pd.DataFrame()
    Modified_Load = pd.DataFrame(columns=["MAC"+"X" + str(mtsp) for mtsp in range(1, mtstpMax)])
    Cov_vec=pd.DataFrame()
    first_index = LoadV.first_valid_index()
    last_index = LoadV.last_valid_index()
    Cov_vec_data = []  # List to store the data for Cov_vec
    for mtsp in range (1, mtstpMax):
        for st in range (1,math.floor((last_index-first_index)/mtsp)): 
            
            if st%5000 == 0:
                print("st=",st)
            Modified_Load.at[st, "MAC"+"X"+ str(mtsp)] = LoadV[(st - 1) * mtsp:(st) * mtsp].sum()
         
        Cov = Modified_Load["MAC"+"X"+ str(mtsp)].std()/Modified_Load["MAC"+ "X"+str(mtsp)].mean()
        Cov_vec_data.append({"Cov": Cov, "MTSP": mtsp})
    
    #Modified_Load.to_csv('Modified_Load.csv')  
    Cov_vec = pd.DataFrame(Cov_vec_data)      
    return Cov_vec, Modified_Load


results = []  # Empty list to store the DataFrames
aa=0




for Cust in Sorted_Cust['Customers'][:41]:
    Cov_vec = pd.read_csv("C:/Omid/CoV_time_Customer"+str(aa)+'.csv')
    results.append(Cov_vec)   # Append the DataFrame to the list
    aa=aa+1



for Cust in Sorted_Cust['Customers'][41:1000]:
    
    print ("No=", aa, "Customer=", Cust)
    LoadVecor=Loads[Cust]
    Cov_vec, Modified_load = CoVfunctin(LoadVecor)  
    results.append(Cov_vec)   # Append the DataFrame to the list
    aa=aa+1
    if aa < 220:
        Modified_load.to_csv("Modified_Load24h"+ str(aa)+".csv")  
    











font = FontProperties()
font.set_family('serif')
font.set_name('Arial')
font.set_size(12)
font.set_weight('bold')

data_Vis=pd.concat([results[i] for i in range (0,len(results))])
data_Vis=data_Vis.reset_index()
data_Vis["MTSP"]=data_Vis["MTSP"]
data_Vis["EUU"]=1/(0.5*data_Vis["MTSP"])
data_Vis.to_csv('data_Vis.csv')  


        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=data_Vis, x="EUU", y="Cov", ci="sd", err_style="band", color='blue')  # Create the line plot




plt.xlabel("Temporal resulotion",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.ylabel("CoV",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.yticks(fontsize=12)  
#plt.xlim(1,maxncing)
plt.xlim(2/(mtstpMax-1), 2)
plt.xticks(fontsize=12)  


# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed

plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y 

plt.savefig('CoV_time_res.png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot


####################################



data_Vis["MTSPm"]=data_Vis["MTSP"]*30

        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=data_Vis, x="MTSPm", y="Cov", ci="sd", err_style="band", color='blue')  # Create the line plot




plt.xlabel("Time step",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.ylabel("CoV",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.yticks(fontsize=12)  
#plt.xlim(1,maxncing)
plt.xlim(30,(mtstpMax-1)*30)
plt.xticks(fontsize=12)  


# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed

plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y 

plt.savefig('CoV_time_step.png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot