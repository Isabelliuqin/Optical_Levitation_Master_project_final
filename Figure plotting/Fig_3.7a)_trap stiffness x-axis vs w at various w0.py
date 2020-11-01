# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:04:28 2020

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


import Module_Fx_stiffness_at_parameter as FTAP

import time

integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 200

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

P = 12.03  #optimal power required to levitate at w0 = 0.85um


##############################################################################
#FIG 3.7a) trap stiffness x-axis at radial offset = (0,0) vs w at various w0
##############################################################################

#x-axis: beam radius w
#y-axis: trap stiffness kx at the beam centre

#key function: FTAP.Fx_stiffness_vs_w_plots

rho_0 = [0 , 0]

rho_0x = 0   #no offset

resolution = 0.01 * 10 ** (-6)  #resolution required to find the gradient at a point



w_0 = [1*10**(-6), 2* 10 ** (-6), 5* 10 ** (-6), 10* 10 ** (-6)]

w = np.linspace(0.5 * np.sqrt(2)*rho, 2 * np.sqrt(2)*rho, 100)  #beam radius, number of element can go to 1000





z_R0 = np.pi * w_0[0]**2 / Lambda

z_R1 = np.pi * w_0[1]**2 / Lambda

z_R2 = np.pi * w_0[2]**2 / Lambda

z_R3 = np.pi * w_0[3]**2 / Lambda

grad_Fx0 = FTAP.Fx_stiffness_vs_w_plots(rho_0x,rho_0[0], rho, n_0, n_s, w_0[0], w, z_R0, P, resolution, target = "reflective", integration_method = integration_method, grid_size = grid_size)

grad_Fx1 = FTAP.Fx_stiffness_vs_w_plots(rho_0x,rho_0[0], rho, n_0, n_s, w_0[1], w, z_R1, P, resolution, target = "reflective", integration_method = integration_method, grid_size = grid_size)

grad_Fx2 = FTAP.Fx_stiffness_vs_w_plots(rho_0x,rho_0[0], rho, n_0, n_s, w_0[2], w, z_R2, P, resolution, target = "reflective", integration_method = integration_method, grid_size = grid_size)

grad_Fx3 = FTAP.Fx_stiffness_vs_w_plots(rho_0x,rho_0[0], rho, n_0, n_s, w_0[3], w, z_R3, P, resolution, target = "reflective", integration_method = integration_method, grid_size = grid_size)




plt.figure(13)
plt.plot(w/(np.sqrt(2) * rho), -np.array(grad_Fx0) * 10 ** 4, lw=2, c="c", label="rho_0x = 0")

plt.plot(w/(np.sqrt(2) * rho), -np.array(grad_Fx1) * 10 ** 8, lw=2, c="b", label="w0 = 2um")

plt.plot(w/(np.sqrt(2) * rho), -np.array(grad_Fx2) * 10 ** 8, lw=2, c="g", label="w0 = 5um")

plt.plot(w/(np.sqrt(2) * rho), -np.array(grad_Fx3) * 10 ** 8, lw=2, c="m", label="w0 = 10um")

print ((w/(np.sqrt(2) * rho))[np.argmin(abs(np.array(grad_Fx0)))]) #print the inflection point

print ((w/(np.sqrt(2) * rho))[np.argmin(abs(np.array(grad_Fx1)))]) #print the inflection point

print ((w/(np.sqrt(2) * rho))[np.argmin(abs(np.array(grad_Fx2)))]) #print the inflection point

print ((w/(np.sqrt(2) * rho))[np.argmin(abs(np.array(grad_Fx3)))]) #print the inflection point


new_ticks1 = np.linspace(0.5, 2, 4) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(-4, 4, 5),fontsize=20)

plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0.5))
ax.spines['bottom'].set_position(('data',0))

plt.legend(loc=1,fontsize=16)

plt.xlabel('w/(sqrt(2)rho)',fontsize=20)

plt.ylabel('kx(10^(-4)N/m)',fontsize=20)


plt.title('rho = 30um, w0 = 2um',fontsize=20)
plt.grid()
plt.show()