
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


maxncing=20

font = FontProperties()
font.set_family('serif')
font.set_name('Arial')
font.set_size(12)
font.set_weight('bold')


data_Vis_groups = pd.read_csv("C:/Omid/data_Vis_groups.csv")


        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot
sns.lineplot(data=data_Vis_groups, x="EUU",  y="Cov", errorbar=("pi",95),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis_groups, x="EUU",  y="Cov", errorbar=("pi",75),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis_groups, x="EUU",  y="Cov", errorbar=("pi",50),  err_style="band", color='blue')  # Create the line plot




plt.xlabel("End user resolution",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.ylabel("CoV",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.yticks(fontsize=12)  
#plt.xlim(1,maxncing)
plt.xlim(1/maxncing,1)
plt.xticks(fontsize=12)  


# Customize font properties for axis tick labels
plt.xticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed
plt.yticks(fontname="Arial", fontsize=10, weight="bold")  # Adjust the size as needed

plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y 

plt.savefig('CoV_EndUser_res.png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot


####################################



import matplotlib.ticker as ticker

        
sns.set(style="whitegrid", font="Arial", rc={"font.size": 12})  # Set the style of the plot

sns.lineplot(data=data_Vis_groups, x="ncing",  y="Cov", errorbar=("pi",95),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis_groups, x="ncing",  y="Cov", errorbar=("pi",75),  err_style="band", color='blue')  # Create the line plot
sns.lineplot(data=data_Vis_groups, x="ncing",  y="Cov", errorbar=("pi",50),  err_style="band", color='blue')  # Create the line plot






plt.xlabel("Number of aggregated end users",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
plt.ylabel("CoV",  fontproperties=font)  # Add a label to the x-axis (replace with your label)
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

plt.savefig('CoV_enduser.png',  format="png", dpi=600, bbox_inches='tight')

plt.show()  # Display the plot