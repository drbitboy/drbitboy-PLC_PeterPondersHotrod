# -*- coding: utf-8 -*-
"""
Fit step function using quadratic with slope of 0 at each known value

Created on Sat 09.Nov, 2024

@author: Brian T. Carcich
Latchmoor Services, INC.
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline,PPoly


def quadinterp(rawxs, rawys):
  """
Y-values are assumed to be mostly constant with single discrete steps
between some single pairs, so slope at each data point is always zero

Fit two quadratics (parabolas) for those discrete steps.

'-' and '/' below show the data with linear interpolation at the step
'*' below show what the parabolas look like

                          ***-----------------------
                        ** /
                       * /
                      */
                     *
                   /*
                 / *
               / **
--------------***

"""
  assert len(rawxs) == len(rawys)

  ### Convert one row of data to two rows
  ### I.e. (n,)-shaped data become (2,n,)-shaped data
  xs,Cs = np.vstack((rawxs,rawxs,)), np.vstack((rawys,rawys,))

  ### Write half-steps to second row between adjacent raw X data pairs 
  xs[1,0:-1] = (xs[0,:-1] + xs[0,1:]) / 2.0
  xs[1,-1] = (xs[0,-1] * 2.0) - xs[1,-2]

  ### Write half-steps to second row between adjacent raw Y data pairs 
  ### Call them C i.e. Y = A x**2 + B x + C
  Cs[1,0:-1] = (Cs[0,:-1] + Cs[0,1:]) / 2.0
  Cs[1,-1] = Cs[0,-1]

  ### Calculate second-order coefficients
  ### - Top row will be positive or 0; bottom row will be negative or 0
  As = np.zeros(Cs.shape)
  As[0,:-1] = 2.0 * (Cs[0,1:] - Cs[0,:-1]) / ((xs[0,1:] - xs[0,:-1])**2)
  As[1,:] = 0.0 - As[0,:]

  ### Calculate first-order coefficients
  ### - Top row will be 0
  ### - Bottom row will be positive or 0
  Bs = np.zeros(Cs.shape)
  Bs[1,:-1] = 2.0 * As[0,:-1] * (xs[0,1:] - xs[1,:-1])

  TF = lambda arr: arr.T.flatten()

  ### Return scipy.interpolate.PPoly class to generate interpolated data
  return PPoly(np.vstack((TF(As),TF(Bs),TF(Cs),))[:,:-1], TF(xs)
              ,extrapolate=True
              )


def main():
    """ enter path and file name for csv that has data to use for
        system identification.
        The file must have a header with three columns.
        Time, Control, and Process Variable
        Don't forget to change the delimiter for the ReadCSV function
        The time units are those used in the input file"""

    global aTime, aCO, aPV, control_interp, b   # These don't change after being initialized
    path = sys.argv[1:] and sys.argv[1] or os.path.join("..", "data", "Hotrod.txt")
    df = pd.read_csv(path, sep='\t', header=0)
    aTime = df.to_numpy()[:,0] / 60.0 	        # convert seconds to minutes
    aCO = df.to_numpy()[:,1]                    # control output, 0-100%
    aPV = df.to_numpy()[:,2]                    # process values, temperatures

    ### Spline interpolation
    control_interp = CubicSpline(aTime, aCO, bc_type='natural')	# for dead time

    ### Quadratic interploation; slope=0 at all data points
    qinterp = quadinterp(aTime,aCO)

    ### Calculate ten interpolate X-values per interval
    L = len(qinterp.x) * 10
    pltxs = qinterp.x[0] + ((qinterp.x[-1] - qinterp.x[0])
                             * np.arange(L+1,dtype=np.float64) / L)

    ### Plot raw data, plot interpolated data
    import matplotlib.pyplot as plt
    plt.plot(aTime,aCO,label='Raw Ctl Output')
    plt.plot(pltxs,qinterp(pltxs),'.',markersize=10,label='Quad-interp')
    plt.plot(pltxs,control_interp(pltxs),'.',label='CubicSpline')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
