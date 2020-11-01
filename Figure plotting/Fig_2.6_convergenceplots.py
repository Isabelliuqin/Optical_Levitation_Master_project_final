# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:30:53 2020

@author: liuqi
"""


import scipy as sp
import numpy as np

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pylab as plt
import scipy.integrate as spi
from scipy.integrate import quad
import seaborn
from scipy.integrate import odeint
from scipy.integrate import dblquad
import Will_Module_addwdep as TQ
import Module_table_parameter as MTP
import Module_integration_manually as MIM

import time

##########################################################
# integration method control: Fx, Fy, Fz convergence plots
##########################################################



integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 1000

plt.close('all')

###########################

#Our sphere

g = 9.8 #gravitational acceleration
c = 3 * 10**8

w_0 = 0.85 * 10 ** (-6)  #random

Lambda = 1.064 * 10**(-6)

z_R = np.pi* w_0 ** 2 / Lambda

rho = 30 * 10 ** (-6) # random
 
n_0 = 1

n_s_n = 0.04

k = 7.6097

n_s = n_s_n - k*1j

sig_s = 10.49 * 10 ** 3 * ( (3 ** 3 - 2.25 ** 3) / 3 ** 3 ) #density of sphere in kg/m^3

sig_0 = 0 #density of medium in kg/m^3

m = 4/3 * np.pi * rho ** 3 * ( sig_s - sig_0 )

Permittivity = 8.85 * 10**(-12)

P = 0.5 * c * n_0 * Permittivity    #total power of the LG01 beam, randomly chosen
###################################################################################


##########################################
#Fx convergence over number of grids
##########################################

#x-axis: gridsize
#y-axis: Fx_manual/Fx_builtin

#key function: TQ.F_total_manual_integration
              #TQ.Fx_total
#Force integration results convergence

rho_0 = [0.5*rho, 0]        #radial offset
w = np.sqrt(2)*rho          #beam radius at target bottom


Fx_list = []
timex_list = []

for i in range(10, grid_size, 20): #obtain a list of manual Fx integration results with various gridsize
    
    start_time = time.time()
    
    integration_results = TQ.F_total_manual_integration(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", coordinate = 'x', grid_size = i)
            
    Fx = integration_results['force_total']
    
    Fx_list.append(Fx)
    
    
    
    timex_list.append(time.time() - start_time) #runningtime list
    print (i)
    
print(Fx_list)

print(timex_list)

start_timeb = time.time()

builtinx=TQ.Fx_total(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective")[0] * 10 ** 12   #builtin force integration result

builintime = time.time() - start_timeb  #builtin function runningtime

number_of_grid = range(30, grid_size, 20)    #exclude the first few results with very small gridsize

plt.figure(1)

plt.scatter(number_of_grid, np.asarray(Fx_list[1:]) * 10 ** 12 / builtinx, label="manual", marker = "^",s = 50)

plt.axhline(y=1, color='r', linestyle='-', label = "dblquad")
plt.legend(loc=4,fontsize=20)

plt.xlabel('number of grids',fontsize=20)
plt.ylabel('Fx_manual/Fx_builtin',fontsize=20)
plt.grid()
plt.show()


#Runningtime comparison

plt.figure(2)
plt.scatter(number_of_grid, np.asarray(timex_list[1:]), label="manual",marker = "8",s = 50)

plt.axhline(y = builintime, color='r', linestyle='-', label = "dblquad")

plt.legend(loc=2,fontsize=20)

plt.xlabel('number of grids',fontsize=20)
plt.ylabel('Fx Runningtime(sec)',fontsize=20)
plt.grid()
plt.show()

start_timem = time.time()
builtiny = TQ.Fx_total(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective")[0] * 10 ** 12
end_timem = time.time()
print (end_timem - start_timem) 

start_time = time.time()
manual = TQ.F_total_manual_integration(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", coordinate = 'x', grid_size = 400)
end_time = time.time()
print (end_time - start_time) 




################################################
#Fy convergence
################################################

#x-axis: gridsize
#y-axis: Fy_manual/Fy_builtin

#key function: TQ.F_total_manual_integration
              #TQ.Fx_total

#Force integration results convergence

rho_0 = [0, 0.5*rho]            #radial offset
w = np.sqrt(2)*rho              #beam radius at target bottom


Fy_list = []
timey_list = []

for i in range(10, grid_size, 20):   #obtain a list of manual Fy integration results with various gridsize
    
    start_time = time.time()
    
    integration_results = TQ.F_total_manual_integration(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", coordinate = 'y', grid_size = i)
            
    Fy = integration_results['force_total']
    
    Fy_list.append(Fy)
    
    
    
    timey_list.append(time.time() - start_time)   #runningtime list
    print (i)
    
print(Fy_list)

print(timey_list)

start_timeb = time.time()

builtiny = TQ.Fy_total(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective")[0] * 10 ** 12    #builtin force integration result

builintime = time.time() - start_timeb    #builtin function runningtime

number_of_grid = range(30, grid_size, 20)   #exclude the first few results with very small gridsize

plt.figure(3)

plt.scatter(number_of_grid, np.asarray(Fy_list[1:]) * 10 ** 12/ builtiny, label="manual", marker = "^", s = 50)

plt.axhline(y=1, color='r', linestyle='-', label = "dblquad")
plt.legend(loc=4,fontsize=20)
plt.ylim(0.995, 1.005)

plt.xlabel('number of grids',fontsize=20)
plt.ylabel('Fy_manual/Fy_builtin',fontsize=20)
plt.grid()
plt.show()



#Runningtime comparison


plt.figure(4)

plt.scatter(number_of_grid, np.asarray(timey_list[1:]), label="manual",marker = "8", s = 50)


plt.axhline(y = builintime, color='r', linestyle='-', label = "dblquad")

plt.legend(loc=2,fontsize=20)

plt.xlabel('number of grids',fontsize=20)
plt.ylabel('Fy Runningtime(sec)',fontsize=20)
plt.grid()
plt.show()






##################################################
#Fz convergence
##################################################

#x-axis: gridsize
#y-axis: Fz_manual/Fz_builtin

#key function: TQ.F_total_manual_integration
              #TQ.Fx_total

#Force integration results convergence


rho_0 = [0,0]                   #radial offset
w = np.sqrt(2)*rho              #beam radius at target bottom


Fz_list = []
time_list = []

for i in range(10, grid_size, 20):       #obtain a list of manual Fy integration results with various gridsize
    
    start_time = time.time()
    
    integration_results =  TQ.F_total_manual_integration(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective", coordinate = 'z', grid_size = i)
            
    Fz = integration_results['force_total']
    
    Fz_list.append(Fz)
    
    
    
    time_list.append(time.time() - start_time)     #runningtime list
    print (i)
    
print(Fz_list)

print(time_list)

start_timeb = time.time()

builtinz = TQ.Fz_total(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, target = "reflective")[0] * 10 ** 12  #builtin force integration result

builintime = time.time() - start_timeb                     #builtin function runningtime

number_of_grid = range(30, grid_size, 20)                  #exclude the first few results with very small gridsize

plt.figure(5)

plt.scatter(number_of_grid, np.asarray(Fz_list[1:]) * 10 ** 12/ builtinz, label="manual", marker = "^",s = 50)

plt.axhline(y=1, color='r', linestyle='-', label ="dblquad")
plt.legend(loc=1,fontsize=20)

plt.xlabel('number of grids',fontsize=20)
plt.ylabel('Fz_manual/Fz_builtin',fontsize=20)
plt.grid()
plt.show()


#Runningtime comparison

plt.figure(6)

plt.scatter(number_of_grid, np.asarray(time_list[1:]), label="manual",marker = "8",s = 50)

plt.axhline(y = builintime, color='r', linestyle='-', label = "dblquad")

plt.legend(loc=2,fontsize=20)

plt.xlabel('number of grids',fontsize=20)
plt.ylabel('Fz Runningtime(sec)',fontsize=20)
plt.grid()
plt.show()





