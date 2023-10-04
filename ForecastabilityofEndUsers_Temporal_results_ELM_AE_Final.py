
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import math
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker


mtstpMax=3

font = FontProperties()
font.set_family('serif')
font.set_name('Arial')
font.set_size(12)
font.set_weight('bold')


data_Vis_2 = pd.read_csv("C:/Omid/Temporal resolution/MAPE_ELMAE_Results_Temporal.csv")






data_Vis=data_Vis_2
#data_Vis["timestep"]=data_Vis["NumberofStepTime"]*30
data_Vis["timeresF"]=1/(data_Vis["NumberofStepTime"]*0.5)

        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=data_Vis, x="timeresF",  y="MAPE", errorbar=("pi",95),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis, x="timeresF",  y="MAPE", errorbar=("pi",75),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis, x="timeresF",  y="MAPE", errorbar=("pi",50),  err_style="band", color='blue')  # Create the line plot




plt.xlabel("Temporal resulotion (1/h)",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.ylabel("MAPE",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.yticks(fontsize=12)  
#plt.xlim(1,maxncing)
plt.xlim(2/(mtstpMax), 2)
plt.ylim(0, 300)

plt.xticks(fontsize=12)  


# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed

plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y 

plt.savefig('MAPE_time_res_ELMA_O_F.png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot


####################################



#data_Vis["MTSPm"]=data_Vis["MTSP"]*30

        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot

sns.lineplot(data=data_Vis, x="timestep",  y="MAPE", errorbar=("pi",95),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis, x="timestep",  y="MAPE", errorbar=("pi",75),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis, x="timestep",  y="MAPE", errorbar=("pi",50),  err_style="band", color='blue')  # Create the line plot






plt.xlabel("Time step (minute)",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.ylabel("MAPE",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.yticks(fontsize=12)  
#plt.xlim(1,maxncing)
plt.xlim(30,(mtstpMax)*30)
plt.ylim(0, 300)

plt.xticks(fontsize=12)  
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))



# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed

plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y 

plt.savefig('MAPE_time_step_ELMA_O_F).png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot