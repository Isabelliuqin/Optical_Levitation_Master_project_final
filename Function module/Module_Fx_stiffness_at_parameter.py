# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:40:03 2020

@author: liuqi
"""


#################################################
#radial stiffness at particular parameter functions
#################################################

import scipy as sp
import numpy as np
import matplotlib.pylab as plt
import scipy.integrate as spi
from scipy.integrate import quad
import seaborn
from scipy.integrate import odeint
from scipy.integrate import dblquad
import cmath
from sympy import *
from scipy.special import eval_genlaguerre as LGpoly

from scipy.misc import derivative
import Will_Module_addwdep as WMTQ


def Fx_stiffness_at_parameter(rho_0x,rho_0y, rho, n_0, n_s, w_0, w, z_R, P, resolution, target, integration_method, grid_size = 500):
    """calculate dFx/drho_0x at a certain rho_0x"""
    
    # rho_0x - beam x displacement, scalar
    # rho_0y - beam y displacement, scalar
    # rho - sphere radius, scalar
    # n_0 - refractive index of medium, scalar
    # n_s - refractive index of sphere, scalar
    # w_0 - beam radius at waist, scalar
    # w - beam radius at target bottom, scalar
    # z_R - beam rayleigh range, scalar
    # P - beam power (Watts), scalar
    # target - string, set as 'reflective' if sphere is reflective, otherwise anything OK and will be ignored
    
    # integration_method - str, 'integrated' or 'manual', use integrated or manual integration method
    # grid_size - integer, number of grids
    # resolution - expansion around the chosen rho_0x, 0.01 * 10 ** (-6)
    
    chosen_resolution = resolution
    rho_0x_minus_2 = rho_0x - 2 * resolution
    rho_0x_minus_1 = rho_0x - 1 * resolution
    
    rho_0x_plus_1 = rho_0x + 1 * resolution
    rho_0x_plus_2 = rho_0x + 2 * resolution
    
    rho_0x_input = [rho_0x_minus_2, rho_0x_minus_1, rho_0x, rho_0x_plus_1, rho_0x_plus_2]
    
    Fx_grad = WMTQ.Fx_total_gradient(rho_0x_input, rho_0y, rho, n_0, n_s, w_0, w, z_R, P, target, integration_method, grid_size)
    Fx_grad_output = np.asarray(Fx_grad['Fx_grad'])
    
    
    return Fx_grad_output

def Fx_stiffness_vs_w0_plots(rho_0x,rho_0y, rho, n_0, n_s, w_0, w, z_R, P, resolution, target, integration_method, grid_size = 500):
    """calculate dFx/drho_0x at a certain rho_0x at different w0 beam waist"""
    
    # rho_0x - beam x displacement, scalar
    # rho_0y - beam y displacement, scalar
    # rho - sphere radius, scalar
    # n_0 - refractive index of medium, scalar
    # n_s - refractive index of sphere, scalar
    # w_0 - beam radius at waist, list
    # w - beam radius at target bottom, scalar
    # z_R - beam rayleigh range, scalar
    # P - beam power (Watts), scalar
    # target - string, set as 'reflective' if sphere is reflective, otherwise anything OK and will be ignored
    
    # integration_method - str, 'integrated' or 'manual', use integrated or manual integration method
    # grid_size - integer, number of grids
    # resolution - expansion around the chosen rho_0x, 0.01 * 10 ** (-6)
    
    Fx_grad = []
    
    Lambda = 1.064 * 10**(-6)
    
    for w0_e in w_0:
        
        z_Re = np.pi* w0_e ** 2 / Lambda
        Fx_grad_element = np.mean(Fx_stiffness_at_parameter(rho_0x,rho_0y, rho, n_0, n_s, w0_e, w, z_Re, P, resolution, target, integration_method, grid_size))
        
        Fx_grad.append(Fx_grad_element)
        
    return Fx_grad



def Fx_stiffness_vs_w_plots(rho_0x,rho_0y, rho, n_0, n_s, w_0, w, z_R, P, resolution, target, integration_method, grid_size = 500):
    'calculation of dFx/drho_0x at a chosen rho_0x vs beam radius w'
    
    # rho_0x - beam x displacement, scalar
    # rho_0y - beam y displacement, scalar
    # rho - sphere radius, scalar
    # n_0 - refractive index of medium, scalar
    # n_s - refractive index of sphere, scalar
    # w_0 - beam radius at waist, scalar
    # w - beam radius at target bottom, list
    # z_R - beam rayleigh range, scalar
    # P - beam power (Watts), scalar
    # target - string, set as 'reflective' if sphere is reflective, otherwise anything OK and will be ignored
    
    # integration_method - str, 'integrated' or 'manual', use integrated or manual integration method
    # grid_size - integer, number of grids
    # resolution - expansion around the chosen rho_0x, 0.01 * 10 ** (-6)
    
    Fx_grad = []
    
    for w_e in w:
        Fx_grad_element = np.mean(Fx_stiffness_at_parameter(rho_0x,rho_0y, rho, n_0, n_s, w_0, w_e, z_R, P, resolution, target, integration_method, grid_size))
        
        Fx_grad.append(Fx_grad_element)
        
    return Fx_grad