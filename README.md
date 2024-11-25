# Second-Order Plus Dead Time modeling

## Usage:

* python SOPDT-GD.py *options ...*

### Command-line options

* **--gradient-descent** - Use gradient descent minimization (default:  Nelder-Mead Simplex)
* **--datapath=Hotrod.txt** - Path to raw data file to use (default:  ../data/Hotrod.txt)
* **--sep=[separator]** - Column separator to use in parsing raw data file (default:  '\\s+' i.e. whitespace)
* **--smooth=N** - Size of moving average of temperature sensor (PV) data
* **--quadinterp** - Use quadratics (parabolas), with slope=0 at segment endpoints, to interpolate Control Output data for dead time (default:  linear interpolation)
* **--cubicinterp** - Use cubic, with slope=0 at segment endpoints to interpolate Control Output data for dead time
* **--splineinterp** - Use cubic spline to interpolate Control Output data for dead time
  * N.B. this option models overshoot and undershoot
* **--xtol=value** - Tolerance for parameter movement per iteration to terminate iterations (default:  1e-06)
* **--ftol=value** - Upper limit for MSE to termination iterations (default:  0.05)
* **--min-alpha=value** - Lower limit for variable alpha (default:  3e-5)
* **--scale-co=value** - Scaling factor to apply to convert raw data file Control Output into range of 0-100% (default:  1.0)
* **--fixedlist=par[,par[,par[,...]]]** - Which model parameter(s), of gain, t0, t1, dt, and off, to fix at their initial value
* **--gain=value** - system gain parameter, degF/%CO (after scaling) (default:  Note 1\*)
* **--t0=value** - time constant 0 to model sensor parameter, minutes (after scaling) (default:  0.685)
* **--t1=value** - time constant 1 parameter, minutes (after scaling) (default:  2.848)
* **--off=value** - ambient temperature parameter, degF (after scaling) (default:  Note 1\*)
* **--dead=value** - deadtime parameter, minutes (after scaling) (default:  0.353)
* **--gain-scale=value** - scale value to multiply\*\* gain parameter (--gain=...) to get model gain value (default:  1.0)
* **--t0-scale=value** - scale value to multiply t0 parameter (--t0=...) to get model t0 value (default:  1.0)
* **--t1-scale=value** - scale value to multiply t1 parameter (--t1=...) to get model t1 value (default:  1.0)
* **--off-scale=value** - scale value to multiply ambient temperature parameter (--off=...) to get model off value (default:  100.0)
* **--dead-scale=value** - scale value to multiply deadtime parameter (--dead=...) to get model deadtime value (default:  1.0)

\* Estimated from raw data, assuming steady states at start and end of data
\*\* Scale values are typically a power of 10 to get all model parameters into the same range
