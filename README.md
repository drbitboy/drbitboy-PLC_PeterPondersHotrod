# Second-Order Plus Dead Time modeling

## Usage:

* python SOPDT-GD.py *options ...*

### Command-line options

* **--datepath=Hotrod.txt** - Path to raw data file to use (default:  ../data/Hotrod.txt)
* **--smooth=N** - Size of moving average of temperature sensor (PV) data
* **--quadinterp** - Use quadratics (parabolas), with slope=0 at segment endpoints, to interpolate Control Output data for dead time (default:  linear interpolation)
* **--cubicinterp** - Use cubic, with slope=0 at segment endpoints to interpolate Control Output data for dead time
* **--splineinterp** - Use cubic spline to interpolate Control Output data for dead time
  * N.B. this option models overshoot and undershoot
* **--xtol=value** - Tolerance for parameter movement per iteration to terminate iterations (default:  1e-06)
* **--fixedlist=par[,par[,par[,...]]]** - Which model parameter(s), of gain, t0, t1, dt, and off, to fix at their initial value
* **--gain=value** - system gain, degF/%CO (default:  Note 1\*)
* **--t0=value** - time constant 0 to model sensor, minutes (default:  0.685)
* **--t1=value** - time constant 1, minutes (default:  2.848)
* **--off=value** - ambient temperature, hdegF\*\* (default:  Note 1\*)
* **--dead=value** - deadtime, minutes (default:  0.353)

\* Estimated from raw data, assuming steady states at start and end of data
\*\* hecto-degrees Fahrenheit i.e. degF / 100.0
