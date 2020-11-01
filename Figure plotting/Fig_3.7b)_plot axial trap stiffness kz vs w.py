

# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 19:07:25 2020

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


#######################################################################################
#FIG 3.7b) plot axial trap stiffness kz at rho_0 = (0,0) vs w at various beam waist w0
#######################################################################################

#x-axis: beam radius w
#y-axis: kz trap stiffness at rho_0 = (0,0)

#key function: TQ.Fz_total_gradient

w = np.linspace(w_0, 2*np.sqrt(2) * rho, 100)  #beam radius



rho_0 = [0 , 0]  #no offset

w_0 = [1*10**(-6), 2* 10 ** (-6), 5* 10 ** (-6), 10* 10 ** (-6)] #various beam waists

z_R0 = np.pi * w_0[0]**2 / Lambda   #Rayleigh ranges

z_R1 = np.pi * w_0[1]**2 / Lambda

z_R2 = np.pi * w_0[2]**2 / Lambda

z_R3 = np.pi * w_0[3]**2 / Lambda


grad_z0 = np.asarray(TQ.Fz_total_gradient(rho_0[0], rho_0[1], rho, n_0, n_s, w_0[0], w, z_R0, P, target = 'reflective', integration_method = integration_method, grid_size = grid_size)['Fz_grad'])

grad_z1 = np.asarray(TQ.Fz_total_gradient(rho_0[0], rho_0[1], rho, n_0, n_s, w_0[1], w, z_R1, P, target = 'reflective', integration_method = integration_method, grid_size = grid_size)['Fz_grad'])

grad_z2 = np.asarray(TQ.Fz_total_gradient(rho_0[0], rho_0[1], rho, n_0, n_s, w_0[2], w, z_R2, P, target = 'reflective', integration_method = integration_method, grid_size = grid_size)['Fz_grad'])

grad_z3 = np.asarray(TQ.Fz_total_gradient(rho_0[0], rho_0[1], rho, n_0, n_s, w_0[3], w, z_R3, P, target = 'reflective', integration_method = integration_method, grid_size = grid_size)['Fz_grad'])


rho_0z0 = np.asarray(TQ.Fz_total_gradient(rho_0[0], rho_0[1], rho, n_0, n_s, w_0[0], w, z_R0, P, target = 'reflective', integration_method = integration_method, grid_size = grid_size)['rho_0z'])

rho_0z1 = np.asarray(TQ.Fz_total_gradient(rho_0[0], rho_0[1], rho, n_0, n_s, w_0[1], w, z_R1, P, target = 'reflective', integration_method = integration_method, grid_size = grid_size)['rho_0z'])

rho_0z2 = np.asarray(TQ.Fz_total_gradient(rho_0[0], rho_0[1], rho, n_0, n_s, w_0[2], w, z_R2, P, target = 'reflective', integration_method = integration_method, grid_size = grid_size)['rho_0z'])

rho_0z3 = np.asarray(TQ.Fz_total_gradient(rho_0[0], rho_0[1], rho, n_0, n_s, w_0[3], w, z_R3, P, target = 'reflective', integration_method = integration_method, grid_size = grid_size)['rho_0z'])




w_in0 = w_0[0] * np.sqrt(1 + (rho_0z0/z_R0)**2)

w_in1 = w_0[1] * np.sqrt(1 + (rho_0z1/z_R1)**2)

w_in2 = w_0[2] * np.sqrt(1 + (rho_0z2/z_R2)**2)

w_in3 = w_0[3] * np.sqrt(1 + (rho_0z3/z_R3)**2)


plt.figure(13)
plt.plot( w_in0/(np.sqrt(2) * rho), -grad_z0*10**8, lw=2, c="r", label="w0 = 1um")#
plt.plot( w_in1/(np.sqrt(2) * rho), -grad_z1*10**8, lw=2, c="b", label="w0 = 2um")#
plt.plot( w_in2/(np.sqrt(2) * rho), -grad_z2*10**8, lw=2, c="g", label="w0 = 5um")#
plt.plot( w_in3/(np.sqrt(2) * rho), -grad_z3*10**8, lw=2, c="m", label="w0= 10um")#

new_ticks1 = np.linspace(0, 2, 5) # plot axis
print(new_ticks1)
plt.xticks(new_ticks1,fontsize=20)
plt.yticks(np.linspace(2, -16, 10),fontsize=20)
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
ax = plt.gca()
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.spines['bottom'].set_position(('data',0))

plt.legend(loc=4,fontsize=16)

#plt.xlabel('w/(sqrt(2)rho)',fontsize=20)
plt.ylabel('kz(10^(-8)N/m)',fontsize=20)

plt.title('rho = 30um, w0 = 2um',fontsize=20)
plt.grid()
plt.show()


'''

'''
