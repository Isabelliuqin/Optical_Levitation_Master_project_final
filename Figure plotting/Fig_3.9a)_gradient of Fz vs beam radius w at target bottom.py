# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 12:54:23 2020

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

import Module_Fz_stiffness_at_parameter as MFSAP

import time



integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 200

plt.close('all')

###########################

#Our sphere

g = 9.8

c = 3 * 10**8

#w_0 = 2 * 10 ** (-6)
w_0 = 0.85 * 10 ** (-6)  #optimal w0 for stiffness matching

Lambda = 1.064 * 10**(-6)

z_R = np.pi* w_0 ** 2 / Lambda

rho = 30 * 10 ** (-6)
 
weight = 3.4 * 10**(-10)

n_0 = 1

n_s_n = 0.04

k = 7.6097

n_s = n_s_n - k*1j

sig_s = 10.49 * 10 ** 3 * (( 3 ** 3 - 2.25 ** 3) / 3 ** 3 ) #density of sphere in kg/m^3

sig_0 = 0 #density of medium in kg/m^3

m = 4/3 * np.pi * rho ** 3 * ( sig_s - sig_0 )

Permittivity = 8.85 * 10**(-12)

#P = 0.5 * c * n_0 * Permittivity    #total power of the LG01 beam

P = 12.03    #optimal power for stiffness matching and stable equilibrium



#########################################################################################
#FIG 3.9a) gradient of Fz at rho_0x/(w/sqrt(2)) =  0.5 vs beam radius w at target bottom
#########################################################################################

#x-axis: beam radius w
#y-axis: gradient of Fz at rho_0x/(w/sqrt(2)) =  0.5

#key functions: MFSAP.Fz_rho_0x_stiffness_at_parameter, TQ.Fz_total_gradient_vs_radialoffset

rho_0 = [0 , 0]

resolution = 0.01 * 10 ** (-6)           #gradient resolution needed by MFSAP.Fz_rho_0x_stiffness_at_parameter

w = np.linspace(0.6 * np.sqrt(2)*rho, 1.5 * np.sqrt(2)*rho, 1000) #target radius, list length can go to 1000

grad_Fzlist = []

for we in w:

    rho_0x_in = 0.5 * we / np.sqrt(2)

    grad_Fz = np.mean(MFSAP.Fz_rho_0x_stiffness_at_parameter(rho_0x_in, rho_0[1], rho, n_0, n_s, w_0, we, z_R, P, resolution, target = 'reflective', integration_method = integration_method, grid_size = grid_size))

    grad_Fzlist.append(grad_Fz*10**4)

plt.figure(13)
plt.plot(w/(np.sqrt(2) * rho), grad_Fzlist, lw=2, c="c", label="rho_0x/(w/sqrt(2)) = 0.5")

print ((w/(np.sqrt(2) * rho))[np.argmin(abs(np.array(grad_Fzlist)))]) #print the inflection point

new_ticks1 = np.linspace(0.6, 1.5, 10) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(-3, 2, 6),fontsize=20)

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0.6))
ax.spines['bottom'].set_position(('data',0))

plt.legend(loc=1,fontsize=16)

plt.xlabel('w/(sqrt(2)rho)',fontsize=20)
plt.ylabel('grad_Fz(10^(-4)N/m)',fontsize=20)

plt.title('rho = 30um, w0 = 0.85um',fontsize=20)
plt.grid()
plt.show()