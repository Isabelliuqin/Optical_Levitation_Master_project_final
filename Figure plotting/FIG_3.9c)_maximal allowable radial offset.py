# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 16:31:57 2020

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



integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 100

plt.close('all')

###########################

#Our sphere

g = 9.8 

c = 3 * 10**8

w_0 = 0.85 * 10 ** (-6)

Lambda = 1.064 * 10**(-6)

z_R = np.pi* w_0 ** 2 / Lambda

rho = 30 * 10 ** (-6)
 
n_0 = 1

n_s_n = 0.04

k = 7.6097

n_s = n_s_n - k*1j

sig_s = 10.49 * 10 ** 3 * (( 3 ** 3 - 2.25 ** 3) / 3 ** 3 ) #density of sphere in kg/m^3

sig_0 = 0 #density of medium in kg/m^3

m = 4/3 * np.pi * rho ** 3 * ( sig_s - sig_0 )

Permittivity = 8.85 * 10**(-12)

#P = 0.5 * c * n_0 * Permittivity    #total power of the LG01 beam

P = 12  #optimal power required to levitate at w0 = 0.85um


############################################
#FIG 3.9c) maximal allowable radial offset
############################################

#x-axis: x-axis radial offset
#y-axis: Qx trapping efficiency

#key function: TQ.F_total_manual_integration

rho_0x = np.linspace(0,2*rho,100)
rho_0 = [0,0]

w = np.sqrt(2) * rho  #optimal beam radius 

Qoplist = []
for rho_0xe in rho_0x:
    
    F_op = TQ.F_total_manual_integration(rho_0xe,rho_0[1], rho, n_0, n_s, w_0, w, z_R, P , target = "reflective", coordinate = 'x', grid_size = grid_size)['force_total'] #compute Qx at optimal beam radius wop, various radial offsets

    Q_op = F_op  * c / ( n_0 * P )
    
    Qoplist.append(Q_op)
    

plt.plot(rho_0x/rho, np.array(Qoplist), lw=2, c="c", label="w/(sqrt(2)rho) = 1")

print ((rho_0x/rho)[np.argmin(abs(np.array(Qoplist)))]) #print the inflection point


new_ticks1 = np.linspace(0, 2 , 5) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(-0.1, 0.05, 4),fontsize=20)

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data',0))

plt.legend(loc=1,fontsize=16)

plt.xlabel('rho_0x/rho',fontsize=20)
plt.ylabel('Qx',fontsize=20)

plt.title('rho = 30um, w0 = 0.85um',fontsize=20)
plt.grid()
plt.show()