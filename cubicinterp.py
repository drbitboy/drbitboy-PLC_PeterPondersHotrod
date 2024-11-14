# -*- coding: utf-8 -*-
"""
Fit step function using cubic with slope of 0 at each known value

Created on Tue 12.Nov, 2024

@author: Brian T. Carcich
Latchmoor Services, INC.
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline,PPoly


def cubicinterp(rawxs, rawys):
  """
Y-values are assumed to be mostly constant with single discrete steps
between some single pairs, so slope at each data point is always zero

Fit cubic for those discrete steps.

'-' and '/' below show the data with linear interpolation at the step
'*' and '_' below show what the cubic looks like

                         _***-----------------------
                        *  /
                       * /
                      */
                     *
                   /*
                 / *
               / _*
--------------***

"""
  assert len(rawxs) == len(rawys)

  ### Convert one row of data to two rows
  ### I.e. (n,)-shaped data become (2,n,)-shaped data
  f64 = np.float64
  xs,ys = np.array(rawxs,dtype=f64), np.array(rawys,dtype=f64)

  ### Calculate delta-x and delta-y
  dxs,dys = (xs[1:] - xs[:-1]), (ys[1:] - ys[:-1])

  ### Calculate third- and second-order coefficients
  As = -2.0 * dys / (dxs**3)
  Bs = 3.0 * dys / (dxs**2)

  ### Calculate first- and zeroth-order coefficients
  Cs = np.zeros(As.shape)
  Ds = ys[:-1]

  ### Return scipy.interpolate.PPoly class to generate interpolated data
  return PPoly(np.vstack((As,Bs,Cs,Ds,)), xs
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
    cinterp = cubicinterp(aTime,aCO)

    ### Calculate ten interpolate X-values per interval
    L = len(cinterp.x) * 10
    pltxs = cinterp.x[0] + ((cinterp.x[-1] - cinterp.x[0])
                             * np.arange(L+1,dtype=np.float64) / L)

    ### Plot raw data, plot interpolated data
    import matplotlib.pyplot as plt
    plt.plot(aTime,aCO,label='Raw Ctl Output')
    plt.plot(pltxs,cinterp(pltxs),'.',markersize=10,label='Quad-interp')
    plt.plot(pltxs,control_interp(pltxs),'.',label='CubicSpline')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
