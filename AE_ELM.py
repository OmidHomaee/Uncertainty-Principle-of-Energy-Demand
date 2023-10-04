# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 12:52:03 2022

@author: Arsalan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error





    
    #%%
def AutoEncoder(TrNum, NhidNeuron, InputNum, P_Train, T_Train, MaxStepBefore, price, TrPercent):  

        
    Length = len(price)
    InputData = np.zeros([Length - MaxStepBefore , MaxStepBefore])
        #OutputData =np.zeros([Length - MaxStepBefore , MaxStepAhead])
        
        
    for i in range(0,Length - MaxStepBefore):
        InputData[i] = price[i:MaxStepBefore + i]
            
            
    OutputData = InputData    
            
    InputNum =  InputData.shape[1]
        #OutputNum = OutputData.shape[1] 
    DataNum = InputData.shape[0]
    TrNum = round(DataNum*TrPercent/100)
        
    P_Train = InputData[0:TrNum]
    T_Train = OutputData[0:TrNum]
        
   
    
    H = np.zeros([TrNum,NhidNeuron])
    a = -1
    b = 1
    
    w1 = np.random.uniform(a,b,[InputNum, NhidNeuron])
    Bias = np.random.uniform(a,b,[InputNum, NhidNeuron])
    
    tempH0 = np.dot(P_Train, w1)
    
    BiasMatrix = np.dot(np.ones([TrNum, InputNum]),Bias)
    
    
    tempH = tempH0 + BiasMatrix
    
    H = np.sin(tempH)
    pinv_H = np.linalg.pinv(H)
    
    w2 = np.dot(pinv_H, T_Train)
    
    return w2
