import sys
import numpy as np
import matplotlib.pyplot as plt
searchlog = (['searchE.log']
             + [s for s in sys.argv[1:]
                if s.startswith('search') and s.endswith('.log')]
            ).pop()
indices = dict(zip('mse fxtol gain t0 t1 off dt'.split(),range(7)))
indices.update(dead=indices['dt'],MSE=indices['mse'])
names = [s for s in sys.argv[1:] if s in indices]

with open(searchlog,'r') as f:
  a=np.array([[float(lst[i]) for i in [2,5,6,7,8,9,10]]
              for lst in
              [L.split() for L in f
               if L.startswith('MSE = ') and ' fxtol = ' in L
              ]
             ]
            )

for ordinal,name in enumerate(names):
  index = indices[name]
  plt.plot(a[:,index])
  plt.title(name)
  if 'fxtol' == name: plt.semilogy()
  plt.show()
  for otherordinal,othername in enumerate(names):
    otherindex = indices[othername]
    if otherordinal <= ordinal: continue
    plt.plot(a[:,index],a[:,otherindex])
    plt.title(f'{name} vs. {othername}')
    if 'fxtol' == othername: plt.semilogy()
    plt.show()
