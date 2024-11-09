import os
import matplotlib.pyplot as plt


def tempplot(aTimes, aPV, aEV, aCO, **kwargs): 
    """ plot the SOPDT response
        aTimes is the array of time values at which PV and CO data was taken
        aPV is the array of provided process variable
        aEV is the array of estimated process variable
        aCO is the array of provieded control output
        block      default false,  Determines if the plot will stays visibe or disappear
        plot_size  defaut 9.0x6.0 inches
        plot_name  default "temperature"
        plot_dir   default "os.getcwd"
        plot_type  default "" options ".svg" ".png" ".pdf{
        plot_dpi   default 100
        """
    _fig, _ax0 = plt.subplots()
    plot_size = kwargs.get("plot_size",(9.0,6.0))
    _fig.set_size_inches(plot_size)
    _line0, = _ax0.plot(aTimes, aPV, 'c-', label='process value')
    _line1, = _ax0.plot(aTimes, aEV, 'r--', label='estimated value')
    _ax0.set_title('Process and Estimated Values vs Time')
    _ax0.set_xlabel('time')             # units are data dependent 
    _ax0.set_ylabel('process and estimated values')

    _ax2 = _ax0.twinx()
    _line2, = _ax2.plot(aTimes, aCO ,'g-',label='control %')
    _ax2.set_ylabel('control %')
    if min(aCO) < 0.0:
        _ax2.set_ylim(-100.0, 100.) 
    else:        
        _ax2.set_ylim(0.0, 100.) 
    
    _lines = [_line0, _line1, _line2]
    _ax0.legend(_lines, [l.get_label() for l in _lines], loc='best')
    _fig.tight_layout()

    model = kwargs.get("model", None)
    if model != None:               # convert to using a dictionary
        k = model[0]
        text =  fr"$k\; = {k:.2f} ^\circ/\%CO$"
        t0 = model[1]
        text += "\n"+fr"$t0 = {t0:.2f}\; min$"
        if len(model) == 3:
            dt = model[2]
            text += "\n"+fr"$dt = {dt:.2f}\; min$"
        if len(model) == 4:
            t1 = model[2]
            text += "\n"+fr"$t1 = {t1:.2f}\; min$"
            dt = model[3]
            text += "\n"+fr"$dt = {dt:.2f}\; min$"
        _ax0.text(0.84, 0.04, text, transform=_ax0.transAxes,
            bbox={'facecolor':'white', 'pad':4, 'alpha':0.7}, family='monospace')
                
    _ax0.grid(True)
    plot_type = kwargs.get("plot_type", "")
    if plot_type:
        plot_dpi = kwargs.get("plot_dpi",100)
        plot_dir = kwargs.get("plot_dir",os.getcwd())
        plot_name = kwargs.get("plot_name","temperature")
        full_plot_name = plot_dir+os.sep+plot_name+plot_type
        plt.savefig(full_plot_name, dpi=plot_dpi)
    plt.show(block=kwargs.get('block', False))

if __name__ == "__main__":
    """ test code  """
    import numpy as np
    import json_subs
    _dict = json_subs.load_dict("test.json")
    aTime = np.array(_dict['lTime'])
    aCO = np.array(_dict['lCO'])
    aPV = np.array(_dict['lPV'])
    aEV = np.array(_dict['lEV'])

    # temperature plot
    tempplot(aTime, aPV, aEV[:,0], aCO, plot_type=".png", block=True)
