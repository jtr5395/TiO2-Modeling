
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


# In[70]:

# Simple first order, constant T, constant P

A = 8.26e4 # s^-1
Ea = 88000 # J/mol
R = 8.314 # J/mol*K
T = 1273 # K

k = A*np.exp(-Ea/(R*T))

# P will not be defined so that the reactant concentrations can be altered directly

end_time = 1

def rates(variables, time):
    """
    Return the right hand side of the ODE
    """
    TiCl4, O2 = variables
    rate_TiCl4 = -k*TiCl4*O2
    rate_O2 = -k*TiCl4*O2
    return (rate_TiCl4, rate_O2)

times = np.linspace(0, end_time, 1000)
initial_conditions = (0.2, 0.1)
result = odeint(rates, initial_conditions, times)
TiCl4 = result[:,0]             
O2 = result[:,1]
plt.plot(times, TiCl4, label='TiCl4')
plt.plot(times, O2, label='O2')
plt.legend(loc="best") # put the legend at the best location to avoid overlapping things
plt.show()


# In[71]:

# Actual rate equation, constant T, constant P

A1 = 8.26e4 # s^-1
A2 = 1.4e5 # 
Ea = 88000 # J/mol
R = 8.314 # J/mol*K
T = 1273 # K

k1 = A1*np.exp(-Ea/(R*T))

k2 = A2*np.exp(-Ea/(R*T))

# P will not be defined so that the reactant concentrations can be altered directly

end_time = 1

def rates(variables, time):
    """
    Return the right hand side of the ODE
    """
    
    TiCl4, O2 = variables
    rate_TiCl4 = -(k1+k2*np.sqrt(O2))*TiCl4
    rate_O2 = -(k1+k2*np.sqrt(O2))*TiCl4
    if (O2 - rate_O2) < 0:
        print(O2 + rate_O2)
        rate_TiCl4 = 0
        rate_O2 = 0
    return (rate_TiCl4, rate_O2)

times = np.linspace(0, end_time, 1000)
initial_conditions = (0.2, 0.21)
result = odeint(rates, initial_conditions, times)
TiCl4 = result[:,0]             
O2 = result[:,1]
plt.plot(times, TiCl4, label='TiCl4')
plt.plot(times, O2, label='O2')
plt.legend(loc="best") # put the legend at the best location to avoid overlapping things
plt.show()


# In[ ]:



