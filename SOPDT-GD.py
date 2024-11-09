# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 13:59:32 2017

@author: Peter Nachtwey
Delta Motion, Inc.
"""
import os
import time
import copy
import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.interpolate import CubicSpline
from tempplot import tempplot           # my script found in the PYTHONPATH



ftol = 0.2                              # search until mse < ftol, it takes a day
xtol = 1e-6                             # distance tolerence, earch until xftol < xtol
alpha = 200e-6                          # learning rate
h = 1e-5                                # finite difference step size



p_enum = (gain, t0, t1, off, dead) = range(0,5) # enumerated global constants, perhaps I should use e_gain instead of gain



def print_params(p):
    print(f"\ngain = {p[gain]:9.6f}")
    print(f"t0   = {p[t0]:9.6f}")
    print(f"t1   = {p[t1]:9.6f}")
    print(f"off  = {p[off]:9.6f}")
    print(f"dead = {p[dead]:9.6f}\n")



def calc_PID(p):
    """ calculate the controller ISA PID parameters """
    _k = p[gain]                        # open loop extend gain
    _t0 = p[t0]
    _t1 = p[t1]
    _c = p[off]                         # output offset or bias
    _dt = p[dead]
    _tc = max(0.1*max(_t0,_t1),0.8*_dt) # closed loop time constant
    _kc = (_t0+_t1)/(_k*(_tc+_dt))      # controller gain %CO/error
    _ti = _t0+_t1                       # integrator time constant
    _td = _t0*_t1/(_t0+_t1)             # derivative time constant
    print(f'\nThe closed loop time constant = {_tc:7.3f} minutes')
    print(f'The controller gain           = {_kc:7.3f} %CO/unit of error')
    print(f'The integrator time constant  = {_ti:7.3f} minutes')
    print(f'The derivative time constant  = {_td:7.3f} minutes')



def init_params():
    p0 = np.empty(len(p_enum))

    p0[gain] = 4.0                      # plant gain deg/% control
    p0[t0] = 0.5                        # time constant 0, min
    p0[t1] = 3.0                        # time constant 1, min
    p0[off] = 70.0                      # ambienmt temperature, deg F
    p0[dead] = 1.0                      # dead time, min

    # bounds
    b = np.array(
        [[0.001,np.inf],                # gain deg/%
        [0.001,np.inf],                 # t0 minutes
        [0.001,np.inf],                 # t1 minutes
        [0.0,np.inf],                   # offset, should be the ambient temperature
        [0.0,np.inf]])                  # dead time minutes
    return p0, b



def difeq(y, t, p):
    """ generate estimated SOPDT solution
        y[0] = process value
        y[1] = rate of change of the process value"""
    _k = p[gain]                        # open loop extend gain
    _t0 = p[t0]
    _t1 = p[t1]
    _c = p[off]                         # output offset or bias
    _dt = p[dead]
    _t = t - _dt
    if _t < 0:                          # don't assume CO befor t=0 is 0
        _u = aCO[0]
    else:
        _u = float(control_interp(max(_t,0)))   # compensate for dead time
    _dy2dt = (-(_t0+_t1)*y[1]-y[0]+_k*_u+_c)/(_t0*_t1)
    return np.array([y[1], _dy2dt])     # return PV' and PV''



def t0p2(p):
    """ use the parameter in p to calculate esimated response
        p is an array of parameters
        aTime is an array of times
        aPV is an array of process variables.
        returns the sum of squared error between the estaimated value
        and the process value  """
    _pv0 = np.array([aPV[0], (aPV[1]-aPV[0])/(aTime[1]-aTime[0])])
    _aEV = odeint(difeq, _pv0, aTime, args=(p,))
    _mse = np.square(np.subtract(aPV,_aEV[:,0])).mean() # mean squared error
    return _mse



def del_f(f, p):
    """ this function calculates the gradient of the cost function at point p
        a small step in the positive direction and negative direction to calculate
        This version calls task() once for each of the parameter to compute the derivative
        for each of the parameters one at a time.
        f computes the cost function using p
        p is an array of the 15 parameters to optimize
        args = (aTime, aActPos) These don't change
    """
    dp = np.zeros_like(p)
    for i in range(len(p)):
        _save = p[i]                    # save p[i] so it can be restored later
        hmax = np.maximum(_save,1.)*h
        p[i] = _save + hmax             # take a step in the positive direction
        fpos = f(p)                     # evaluate after taking a positive step
        p[i] = _save - hmax             # take a step in the negative direction.
        fneg = f(p)                     # evaluate after taking a negative step
        p[i] = _save                    # restore cell to its original value
        dp[i] = (fpos-fneg)/(2.*hmax)   # calculate the gradient with p[i] as the center value
    return dp                           # return the gradient



def gradient_descent(f, p):
    """ minimize the cost function f using parameters p
    """
    mse = f(p)
    print(f'MSE = {mse:12.9f}')                 # initial mean squared error
    fxtol = 1.01*xtol                           # filter distance tolerence
    while mse > ftol and fxtol > xtol:          # test for change in parameters
        grad = del_f(f,p)                       # gradient
        gnorm = np.linalg.norm(grad)            # gradient norm
        step = -alpha*grad                      # the step is opposite
        p += step                               # update the parameters
        p = np.fmax(np.fmin(p,b[:,1]),b[:,0]) 	# bounds check
        _mse = f(p)                             # cost function
        if _mse < mse:
            fxtol += 0.2*(np.linalg.norm(step/p)-fxtol) # beware of divide by 0!
            mse = _mse
            # print(f'MSE = {_mse:12.9f} gnorm = {gnorm:6.3f} {p[0]:8.6f} {p[1]:8.6f} {p[2]:8.6f} {p[3]:8.6f} {p[4]:8.6f}')
            print(f'MSE = {_mse:12.9f} fxtol = {fxtol:.3E} {p[0]:8.6f} {p[1]:8.6f} {p[2]:8.6f} {p[3]:8.6f} {p[4]:8.6f}')
    return p, mse



def main():
    """ enter path and file name for csv that has data to use for
        system identification.
        The file must have a header with three columns.
        Time, Control, and Process Variable
        Don't forget to change the deliminator for the ReadCSV function
        The time units are those used in the input file"""

    global aTime, aCO, aPV, control_interp, b   # These don't change after being initialized
    path = os.path.join("..", "data", "Hotrod.txt")
    df = pd.read_csv(path, sep='\t', header=0)
    aTime = df.to_numpy()[:,0]
    aCO = df.to_numpy()[:,1]                    # control output, 0-100%
    aPV = df.to_numpy()[:,2]                    # process values, temperatures
    aTime /= 60.0		    	                # convert seconds to minutes
    control_interp = CubicSpline(aTime, aCO, bc_type='natural')	# for dead time

    p0, b = init_params()                       # initial parameters and bounds
    time0 = time.process_time()
    p_opt, mse = gradient_descent(t0p2, p0)     # p_opt are the optimizined parameters
    diftime = time.process_time()-time0
    m, s = divmod(int(diftime),60)
    h, m = divmod(m,60)
    print(f"\nElasped Time = {h:02d}:{m:02d}:{s:02d}")
    dp = del_f(t0p2, p_opt)                     # the gradient at the 'minimum'
    gnorm = np.linalg.norm(dp)                  # the gradient norm at the 'minimum'
    print(f'\nMSE = {mse:12.9f}  RMSE = {np.sqrt(mse):12.9f} gnorm = {gnorm:.3f}')

    print_params(p_opt)
    _k = p_opt[gain]                            # open loop gain.  PV change / %control output
    _t0 = p_opt[t0]                             # time constant 0
    _t1 = p_opt[t1]                             # time constant 1
    _c = p_opt[off]                             # PV offset, ambient PV
    _dt = p_opt[dead]                           # dead time
    calc_PID(p_opt)                             # use optimize parameters for calculating PID

    # initial process value and rate of change
    pv0 = np.array([aPV[0], (aPV[1]-aPV[0])/(aTime[1]-aTime[0])])
    # show the response for the estimated system.
    aEV = odeint(difeq, pv0, aTime, args=(p_opt,))

    # save for debugging
    import json
    lTime = aTime.tolist()
    lCO = aCO.tolist()
    lPV = aPV.tolist()
    lEV = aEV.tolist()
    dict_of_lists = {'lTime':lTime, 'lCO':lCO, 'lPV':lPV, 'lEV':lEV}
    with open("SOPDT_GD.json", 'w') as f: json.dump(dict_of_lists, f)

    # temperature plot
    plot_name = os.path.basename(__file__).split('.',1)[0]
    tempplot(aTime, aPV, aEV[:,0], aCO, plot_name = plot_name, plot_type=".png", model=(_k,_t0,_t1,_dt), block=True)



if __name__ == "__main__":
    main()

"""

UM690

ftol = 0.2                              # search until mse < ftol, it takes a day
xtol = 1e-6                             # distance tolerence, earch until xftol < xtol
alpha = 200e-6                          # learning rate
h = 1e-5                                # finite difference step size




ftol = 0.2                              # search until mse < ftol, it takes a day
xtol = 1e-8                             # distance tolerence, earch until xftol < xtol
alpha = 200e-6                          # learning rate
h = 1e-5                                # finite difference step size




Z-Box

mse_tol = 0.3               # search until mse < mse_tol
alpha = 20e-6               # learning rate
h = 1e-9                    # step size

Elasped Time = 00:14:06

mse = 0.2970118558  gnorm= 51.170268

gain =  3.848170
t0   =  0.709176
t1   =  2.852310
off  = 70.645262
dead =  0.293374

The closed loop time constant =   0.285 minutes
The controller gain           =   1.600 %CO/unit of error
The integrator time constant  =   3.561 minutes
The derivative time constant  =   0.568 minutes


Zen

mse_tol = 0.3                           # search until mse < mse_tol
alpha = 200e-6                          # learning rate
h = 1e-6                                # for finite difference

Elasped Time = 00:10:24

MSE =  0.299999995  gnorm= 0.034

gain =  3.852292
t0   =  0.728601
t1   =  2.835242
off  = 70.208479
dead =  0.284216


The closed loop time constant =   0.284 minutes
The controller gain           =   1.629 %CO/unit of error
The integrator time constant  =   3.564 minutes
The derivative time constant  =   0.580 minutes


mse_tol = 0.3               # search until mse < mse_tol
alpha = 20e-6               # learning rate
h = 1e-9                    # step size

Elasped Time = 00:07:32

mse = 0.2998668665  gnorm= 76.187132

gain =  3.850499
t0   =  0.718030
t1   =  2.839879
off  = 70.354910
dead =  0.291508

The closed loop time constant =   0.284 minutes
The controller gain           =   1.606 %CO/unit of error
The integrator time constant  =   3.558 minutes
The derivative time constant  =   0.573 minutes

mse_tol = 0.3               # search until mse < mse_tol
alpha = 20e-6               # learning rate
h = 1e-9                    # step size


Elasped Time = 00:21:12

mse = 0.2999540523  gnorm= 57.376507

gain =  3.851035
t0   =  0.730017
t1   =  2.833757
off  = 70.306055
dead =  0.283399

The closed loop time constant =   0.283 minutes
The controller gain           =   1.633 %CO/unit of error
The integrator time constant  =   3.564 minutes
The derivative time constant  =   0.580 minutes



"""
