# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 22:47:10 2020

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

############################
# integration method control
############################

integration_method = 'manual'   # 'manual' or 'integrated'
grid_size = 400

plt.close('all')

###########################


g = 9.8 

c = 3 * 10**8

w_0 = 0.85 * 10**(-6)


Lambda = 1.064 * 10**(-6)

z_R = np.pi* w_0 ** 2 / Lambda

rho = 30 * 10 ** (-6)
 
n_0 = 1

n_s_n = 0.04

k = 7.6097

n_s = n_s_n - k*1j

sig_s = 10.49 * 10 ** 3 * (( 3 ** 3 - 2.25 ** 3) / 3 ** 3 ) #density of sphere in kg/m^3

sig_0 = 0                                                   #density of medium in kg/m^3

m = 4/3 * np.pi * rho ** 3 * ( sig_s - sig_0 )              #mass of sphere

Permittivity = 8.85 * 10**(-12)

P_norm = 0.5 * c * n_0 * Permittivity   

               


rho_0 = [0,0]

F = TQ.F_total_manual_integration(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, np.sqrt(2) * rho, z_R, P_norm , target = "reflective", coordinate = 'z', grid_size = grid_size)['force_total']

P_op = -m*g*P_norm / F

Fop = TQ.F_total_manual_integration(rho_0[0],rho_0[1], rho, n_0, n_s, w_0, np.sqrt(2) * rho, z_R, P_op , target = "reflective", coordinate = 'z', grid_size = grid_size)['force_total']

diff = Fop + m*g

print (diff)
print (P_op)
