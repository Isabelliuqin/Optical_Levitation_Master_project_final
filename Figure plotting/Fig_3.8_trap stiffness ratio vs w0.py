# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 17:27:40 2020

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


import Module_Fz_stiffness_at_parameter as FTAPz
import Module_Fx_stiffness_at_parameter as FTAPx

import time





integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 200

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

#P = 0.5 * c * n_0 * Permittivity    #total power of the LG01 beam

P = 12.03

###########################################################################
#FIG 3.8 kz/kx vs w0 at optimal beam radius w = sqrt(2)rho vs beam waist w0
###########################################################################

#x-axis: beam waist w0
#y-axis: trap stiffness ratio kz/kx at optimal trapping position rho_0 = (0,0), w =sqrt(2)rho

#key function: FTAPz.Fz_stiffness_vs_w0_plots, FTAPx.Fx_stiffness_vs_w0_plots

a = 30 * 10 ** (-6)

rho = a

w = np.sqrt(2) * rho  #optimal beam radius

rho_0 = [0 , 0]       #optimal radial trapping position

resolution = 0.01 * 10 ** (-6)   #resolution needed to compute force gradient at a point

w_0 = np.linspace(0.5*10**(-6), 0.85 * 10**(-6),100)    #various beam waist


grad_z0 = np.asarray(FTAPz.Fz_stiffness_vs_w0_plots(rho_0[0], rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, resolution, target = 'reflective', integration_method = integration_method, grid_size = grid_size))

grad_x0 = np.asarray(FTAPx.Fx_stiffness_vs_w0_plots(rho_0[0], rho_0[1], rho, n_0, n_s, w_0, w, z_R, P, resolution, target = 'reflective', integration_method = integration_method, grid_size = grid_size))
    
plt.figure(14)
plt.plot( w_0 * 10 ** 6, (-grad_z0*10**4 / (-grad_x0*10**4)), lw=2, c="c", label="kz / kx, w/(sqrt(2)rho) = 1")
print ((w_0 * 10 ** 6)[np.argmin(abs((-grad_z0*10**8 / (-grad_x0*10**8)) - 1))] ) 
plt.axhline(y=1, color='r', linestyle='-', label = "kz / kx = 1")


new_ticks1 = np.linspace(0, 20, 5) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(0, 2, 5),fontsize=20)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)

ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.spines['bottom'].set_position(('data',0))

plt.legend(loc=1,fontsize=16)

plt.xlabel('w0(um)',fontsize=20)
plt.ylabel('kz / kx',fontsize=20)

plt.title('rho = 30um',fontsize=20)
plt.grid()
plt.show()