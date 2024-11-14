# -*- coding: utf-8 -*-
"""
Fit step function using piece-wise linear interpolation

Created on Wed 13.Nov, 2024

@author: Brian T. Carcich
Latchmoor Services, INC.
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline,PPoly
from quadinterp import quadinterp
from cubicinterp import cubicinterp


def linearinterp(rawxs, rawys):
  """
Y-values are assumed to be mostly constant with single discrete steps
between some single pairs, so slope at each data point is always zero

'-' below shows the data
'/' below shows the data with linear interpolation at the step

                     -----------------------
                    /
                   /
                  /
                 /
                /
               / 
              /
--------------

"""
  assert len(rawxs) == len(rawys)

  ### Convert one row of data to two rows
  ### I.e. (n,)-shaped data become (2,n,)-shaped data
  f64 = np.float64
  xs,ys = np.array(rawxs,dtype=f64), np.array(rawys,dtype=f64)

  ### Calculate delta-x and delta-y
  dxs,dys = (xs[1:] - xs[:-1]), (ys[1:] - ys[:-1])

  ### Calculate first- and zeroth-order linear coefficients
  As = dys / dxs
  Bs = ys[:-1]

  ### Return scipy.interpolate.PPoly class to generate interpolated data
  return PPoly(np.vstack((As,Bs,)), xs
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
    aTime = df.to_numpy()[:,0]                  # Leave as seconds here
    aCO = df.to_numpy()[:,1]                    # control output, 0-100%
    aPV = df.to_numpy()[:,2]                    # process values, temperatures

    ### Spline interpolation
    CSinterp = CubicSpline(aTime, aCO, bc_type='natural')

    ### Piece-wise linear interpolation
    linterp = linearinterp(aTime,aCO)

    ### Quadratic and Cubic interpolation; slope=0 at all data points
    qinterp = quadinterp(aTime,aCO)
    cinterp = cubicinterp(aTime,aCO)

    ### Calculate ten interpolated time values per interval
    L = len(cinterp.x) * 10
    pltxs = cinterp.x[0] + ((cinterp.x[-1] - cinterp.x[0])
                             * np.arange(L+1,dtype=np.float64) / L)

    ### Plot raw data, plot interpolated data
    import matplotlib.pyplot as plt
    plt.plot(aTime,aCO,label='Raw Ctl Output')
    plt.plot(pltxs,CSinterp(pltxs),'.',markersize=25,label='CubicSpline')
    plt.plot(pltxs,cinterp(pltxs),'.',markersize=20,label='Cubic-interp')
    plt.plot(pltxs,qinterp(pltxs),'.',markersize=15,label='Quadric-interp')
    plt.plot(pltxs,linterp(pltxs),'.',markersize=10,label='Linear-interp')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
