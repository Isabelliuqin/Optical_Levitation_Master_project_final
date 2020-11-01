# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:59:30 2020

@author: liuqi
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:57:39 2020

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


import time

##########################################################
# Integration control: 3D force contours
##########################################################



integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 400  #appropriate gridsize

plt.close('all')

###########################

#Our sphere

g = 9.8

c = 3 * 10**8

w_0 = 2 * 10 ** (-6)

Lambda = 1.064 * 10**(-6)

z_R = np.pi* w_0 ** 2 / Lambda

rho = 30 * 10 ** (-6)
 
n_0 = 1

n_s_n = 0.04

k = 7.6097

n_s = n_s_n - k*1j

sig_s = 10.49 * 10 ** 3 * ( (3 ** 3 - 2.25 ** 3) / 3 ** 3 ) #density of sphere in kg/m^3

sig_0 = 0 #density of medium in kg/m^3

m = 4/3 * np.pi * rho ** 3 * ( sig_s - sig_0 )

Permittivity = 8.85 * 10**(-12)

P = 0.5 * c * n_0 * Permittivity    #total power of the LG01 beam



####################################################
#Qz vs 2D w and offset full range
####################################################

#x-axis: beam radius at target bottom
#y-axis: radial offset
#Qz,  trapping efficiency

#key functions: TQ.F_total_manual_integration
rho = 30 * 10 ** (-6)        #target radius

w = np.linspace(w_0, 2*np.sqrt(2)*rho, 40)    #beam radius at target radius

rho_0 = [0, 0]              

rho_0[0] = np.linspace(-3*rho, 3*rho, 40)     #x-axis radial offset

w,rho_0x = np.meshgrid(w, rho_0[0])


Fzarray = [ [ None for p in range( len(w) ) ]
               for q in range( len(rho_0x) ) ] 
Qzarray = [ [ None for p in range( len(w) ) ]
               for q in range( len(rho_0x) ) ] 
        
for i in range(len(w)):
            
    for k in range(len(rho_0x)):        


        Fzarray[k][i] = TQ.F_total_manual_integration(rho_0x[k][i],rho_0[1], rho, n_0, n_s, w_0, w[k][i], z_R, P, target = 'reflective', coordinate = 'z', grid_size = grid_size)['force_total']
            
        

        Qzarray[k][i] = Fzarray[k][i] * c / ( n_0 * P )    #REFERENCE: Eq.(2.21)


plt.contourf(w/(np.sqrt(2)*rho), rho_0x/(rho), np.array(Qzarray), 25, cmap='viridis')
cbar = plt.colorbar()

cbar.set_ticks([-2, -1.5, -1, -0.5, 0])
plt.xlabel('w/(sqrt(2)rho)',fontsize=20)
plt.ylabel('rho_0x/rho',fontsize=20)

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)


plt.title('Qz, rho = 30um, w0 = 2um',fontsize=20)
plt.grid()
plt.show() 




####################################################
#Qx vs 2D w and offset full range
####################################################


#x-axis: beam radius at target bottom
#y-axis: radial offset
#Qx,  trapping efficiency

#key functions: TQ.F_total_manual_integration
rho = 30 * 10 ** (-6)    #target radius

w = np.linspace(w_0, 2*np.sqrt(2)*rho, 40)   #beam radius at target radius

rho_0 = [0, 0]

rho_0[0] = np.linspace(-3*rho, 3*rho, 40)    #x-axis radial offset

w,rho_0x = np.meshgrid(w, rho_0[0])


Fxarray = [ [ None for p in range( len(w) ) ]
               for q in range( len(rho_0x) ) ] 
Qxarray = [ [ None for p in range( len(w) ) ]
               for q in range( len(rho_0x) ) ] 


for i in range(len(w)):
            
    for k in range(len(rho_0x)):        


        Fxarray[k][i] = TQ.F_total_manual_integration(rho_0x[k][i],rho_0[1], rho, n_0, n_s, w_0, w[k][i], z_R, P, target = 'reflective', coordinate = 'x', grid_size = grid_size)['force_total']
            
        Qxarray[k][i] = Fxarray[k][i] * c / ( n_0 * P )    #REFERENCE: Eq.(2.21)



plt.contourf(w/(np.sqrt(2)*rho), rho_0x/(rho), np.array(Qxarray), 25, cmap='viridis')
cbar = plt.colorbar()

cbar.set_ticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel('w/(sqrt(2)rho)',fontsize=20)
plt.ylabel('rho_0x/rho',fontsize=20)

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)


plt.title('Qx, rho = 30um, w0 = 2um',fontsize=20)
plt.grid()
plt.show()  




