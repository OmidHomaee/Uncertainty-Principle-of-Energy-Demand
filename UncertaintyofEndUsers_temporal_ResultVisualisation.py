
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


mtstpMax=48

font = FontProperties()
font.set_family('serif')
font.set_name('Arial')
font.set_size(12)
font.set_weight('bold')


data_Vis = pd.read_csv("C:/Omid/data_Vis.csv")


        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=data_Vis, x="EUU",  y="Cov", errorbar=("pi",95),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis, x="EUU",  y="Cov", errorbar=("pi",75),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis, x="EUU",  y="Cov", errorbar=("pi",50),  err_style="band", color='blue')  # Create the line plot




plt.xlabel("Temporal resulotion (1/h)",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
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

plt.savefig('CoV_time_resVer2.png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot


####################################



data_Vis["MTSPm"]=data_Vis["MTSP"]*30

        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot

sns.lineplot(data=data_Vis, x="MTSPm",  y="Cov", errorbar=("pi",95),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis, x="MTSPm",  y="Cov", errorbar=("pi",75),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis, x="MTSPm",  y="Cov", errorbar=("pi",50),  err_style="band", color='blue')  # Create the line plot






plt.xlabel("Time step (minute)",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
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

plt.savefig('CoV_time_stepVer2.png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot