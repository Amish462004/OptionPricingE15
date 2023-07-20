# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 02:05:04 2023

@author: amish
"""

#
# Monte Carlo valuation of European call options with NumPy (log version)
# Monte_Carlo.py
#
import math
from numpy import *
from time import time
# star import for shorter code
random.seed(20000)
t0 = time()
# Parameters
S0 = 100.; SK = 105.; T = 1.0; r = 0.067; sigma = 0.2
M = 50; dt = T / M; I = 250000
# Simulating I paths with M time steps
S = S0 * exp(cumsum((r - 0.5 * sigma ** 2) * dt
+ sigma * math.sqrt(dt)
* random.standard_normal((M + 1, I)), axis=0))                  

# if only the final values are of interest
S[0] = S0
# Calculating the Monte Carlo estimator
# C0 is the dicounted gain from the option
C0 = math.exp(-r * T) * sum(maximum(S[-1] - SK, 0)) / I
# Results output
tnp2 = time() - t0

print('The payoff is ', C0)  
print('The Execution Time is: ',tnp2)  

import matplotlib.pyplot as plt
plt.plot(S[:, :20])
plt.grid(True)
plt.xlabel('Steps')
plt.ylabel('Index level')
plt.show()

plt.rcParams["figure.figsize"] = (15,8)
plt.hist(S[-1], bins=50)
plt.grid(True)
plt.xlabel('index level')
plt.ylabel('frequency')