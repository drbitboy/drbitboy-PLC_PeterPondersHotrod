import sys
import numpy as np
import matplotlib.pyplot as plt
searchlog = (['searchE.log']
             + [s for s in sys.argv[1:]
                if s.startswith('search') and s.endswith('.log')]
            ).pop()
indices = dict(zip('fxtol gain t0 t1 off dt'.split(),range(6)))
names = [s for s in sys.argv[1:] if s in indices]

with open(searchlog,'r') as f:
  a=np.array([list(map(float,L.split()[5:]))
              for L in f
              if L.startswith('MSE = ') and ' fxtol = ' in L
             ]
            )

for name in names:
  index = indices[name]
  plt.plot(a[:,index])
  plt.title(name)
  if 'fxtol' == name: plt.semilogy()
  plt.show()
  for othername in names:
    if othername == name: continue
    otherindex = indices[othername]
    plt.plot(a[:,index],a[:,otherindex])
    plt.title(f'{name} vs. {othername}')
    if 'fxtol' == othername: plt.semilogy()
    plt.show()
