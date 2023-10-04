import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import math

df_loads = pd.read_csv("C:/Omid/archive/halfhourly_dataset/halfhourly_dataset/block_0.csv")
information_housholds = pd.read_csv("C:/Omid/archive/informations_households.csv")

for j in range (1,112):
    print("j=", j)
    df_input= pd.read_csv("C:/Omid/archive/halfhourly_dataset/halfhourly_dataset/block_"+str(j)+".csv")
    df_loads=pd.concat([df_loads,df_input])


#df_loads.to_csv('ALLHHloads.csv')  
CusomersList = pd.DataFrame(columns=["Customers", "Installation"])
Customers=information_housholds[information_housholds["stdorToU"]=="Std"]["LCLid"]
CusomersList['Customers']=Customers
CusomersList=CusomersList.reset_index()

################
#a=0
#for cust in CusomersList["Customers"]:
#    a=a+1
#    if a%10 == 0:   
#        print("a=", a)
#    CusomersList["Installation"][CusomersList["Customers"]==cust] = df_loads['tstp'][df_loads['LCLid']==cust].head(1).values
    
#CusomersList["Installation"]=pd.to_datetime(CusomersList["Installation"])
#Sorted_Cust=CusomersList.sort_values(by='Installation')
#Sorted_Cust=Sorted_Cust.reset_index()
#Sorted_Cust.to_csv('Sorted_Cust.csv')  
##########

Sorted_Cust = pd.read_csv("C:/Omid/Sorted_Cust.csv")

########

Loads=pd.DataFrame 
A=df_loads['tstp'][df_loads['LCLid']==Sorted_Cust["Customers"][0]]
Loads = pd.DataFrame(A, columns=['tstp'])
Loads[Sorted_Cust["Customers"][0]]=pd.to_numeric(df_loads['energy(kWh/hh)'][df_loads['LCLid']==Sorted_Cust["Customers"][0]], errors='coerce') 
Loads.reset_index(inplace = True)


Loads = pd.read_csv("C:/Omid/ModiLoads500.csv")
Cn=499
for i in Sorted_Cust["Customers"][499:]:
    Cn = Cn+1
    print ("Cn=", Cn, i)
    if Cn == 1000:
        Loads.to_csv('ModiLoads1000.csv')  
    if Cn == 2000:
        Loads.to_csv('ModiLoads2000.csv')    
    B=pd.DataFrame()
    B=df_loads[df_loads['LCLid']==i]
    indexa=-1
    for timestep in Loads['tstp']:
        indexa += 1
     #   if indexa%10000 == 0:
      #      print("indexa=",indexa)
        indexb = -1
        indexb=B[B['tstp']== timestep].index.values
        if  indexb != -1:
            Loads.at[Loads.index[indexa], i] = pd.to_numeric(B['energy(kWh/hh)'][indexb].values, errors='coerce')                 


Loads.to_csv('ModiLoadsALL.csv')  


Modified_Load_G={}
results_group = {}  # Empty list to store the DataFrames
Numberofgroups=5
maxncing=600
for groupindex in range (1,Numberofgroups+1):
    Cov_vec_data=[]
    print("Group no=",groupindex)
    Modified_Load_G[groupindex] =pd.DataFrame()
    Modified_Load_G[groupindex] = pd.DataFrame(columns=["G"+str(groupindex)+"X" + str(ncing) for ncing in range(0, maxncing)])
    LoadVecor_Group= Loads.iloc[:, 2+((groupindex-1)*maxncing):2+(groupindex*maxncing)]
    for ncing in range (0,maxncing):
        print("Sub Group no=",ncing)
        Modified_Load_G[groupindex]["G"+str(groupindex)+"X" + str(ncing)]= LoadVecor_Group.iloc[:,0]
        for jj in range (1,ncing+1):
            Modified_Load_G[groupindex]["G"+str(groupindex)+"X" + str(ncing)]= Modified_Load_G[groupindex]["G"+str(groupindex)+"X" + str(ncing)] + LoadVecor_Group.iloc[:,jj]
        Cov = Modified_Load_G[groupindex]["G"+str(groupindex)+"X" + str(ncing)].std()/Modified_Load_G[groupindex]["G"+str(groupindex)+"X" + str(ncing)].mean()
        Cov_vec_data.append({"Cov": Cov, "NG": groupindex, "ncing":ncing})
        Cov_vec = pd.DataFrame(Cov_vec_data)   
        results_group[groupindex]=Cov_vec  # Append the DataFrame to the list   
        results_group[groupindex].to_csv("CoV_G"+str(groupindex)+'.csv')
    Modified_Load_G[groupindex].to_csv("G"+str(groupindex)+'.csv')
    
    
    
    
    
n=-1
for groupindex in range (1,Numberofgroups+1):
    n=n+1
   # df.to_csv('CoV_group'+str(n)+'.csv')  
    # Plot the data from each DataFrame
   # if n !=16:
    plt.plot(results_group[groupindex]["ncing"]+1,results_group[groupindex]["Cov"], label = "Group #"+str(groupindex), linewidth=2.5) 
# Add labels, title, etc. to the plot as needed
plt.xlabel('Number of end users')
plt.ylabel('CoV')
plt.xlim(1, maxncing)
#labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', ]  
#plt.xticks(df["ncing"], labels, rotation = 40, fontsize=12)  
plt.yticks(fontsize=12)  
plt.legend()
  
plt.rc('axes', titlesize=14)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y 
plt.grid()
# Show the plot
plt.savefig('CoV_group_NE.png',  format="png", dpi=600, bbox_inches='tight')
plt.show()


