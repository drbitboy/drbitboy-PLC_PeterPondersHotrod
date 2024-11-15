# Second-Order Plus Dead Time modeling

## Usage:

* python SOPDT-GD.py *options ...*

### Command-line options

* **--datepath=Hotrod.txt** - Path to raw data file to use (default:  ../data/Hotrod.txt)
* **--smooth=N** - Size of moving average of Control Output data
* **--quadinterp** - Use quadratics (parabolas), with slope=0 at segment endpoints, to interpolate Control Output data for dead time (default:  linear interpolation)
* **--cubicinterp** - Use cubic, with slope=0 at segment endpoints to interpolate Control Output data for dead time
* **--splineinterp** - Use cubic spline to interpolate Control Output data for dead time
  * N.B. this option models overshoot and undershoot
* **--xtol=value** - Tolerance for parameter movement per iteration to terminate iterations (default:  1e-06)
