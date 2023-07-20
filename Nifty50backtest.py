# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 00:44:41 2023

@author: amish
"""

import math
import numpy as np
import pandas as pd
from numpy import *
from time import time
import matplotlib.pyplot as plt
# star import for shorter code
random.seed(20000)
t0 = time()
# Parameters
S0 = 18297. #Index Price at step 0 
T = 1.0/50;
r = 0.067 #risk free return rate 
sigma = 0.11 #volatility 
M = 50; dt = T / M; I = 250000
step=M
L=[] #empty list for storing averaged values after each step
# Simulating I paths with M time steps
for i in range(0,step):

  S = S0 * exp(cumsum((r - 0.5 * sigma ** 2) * dt
  + sigma * math.sqrt(dt)
  * random.standard_normal((M + 1, I)), axis=0))
  S[0]=S0
  
  L.append(S[-1].mean())
  S0=S[-1].mean()  #Updating index at each step

print(S0) #Final predicted value of index 

#Importing historic data for backtesting
df=pd.read_csv("^NSEI.csv")
df=df.tail(50)
df.reset_index(drop=True,inplace=True)

#Plotting the historic data and prediced path on same graph
import matplotlib.pyplot as plt
plt.plot(L,label="Predicted path",color='blue')
plt.plot(df['Close'], label="Actual Path",color='green')
plt.grid(True)
plt.xlabel('Days')
plt.ylabel('Nifty 50')
plt.legend()
plt.show()

#calculating RMSE for the data
MSE = np.square(np.subtract(L,df['Close'])).mean() 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error:")
print(RMSE)