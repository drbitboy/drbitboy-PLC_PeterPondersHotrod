import sys
import matplotlib.pyplot as plt
searchlog = (['searchE.log']
             + [s for s in sys.argv[1:]
                if s.startswith('search') and s.endswith('.log')]
            ).pop()
indices = dict(zip('fxtol gain t0 t1 off dt'.split(),range(6)))
name = (['dt']+[s for s in sys.argv[1:] if s in indices]).pop()
index = indices[name] + 5

with open(searchlog,'r') as f:
  a=[float(L.split()[index])
     for L in f
     if L.startswith('MSE = ') and ' fxtol = ' in L
    ]
plt.plot(a)
plt.title(name)
if 'fxtol' == name: plt.semilogy()
plt.show()
