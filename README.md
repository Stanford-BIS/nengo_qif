# nengo_QIF
Contains QIF neuron models for use in nengo

Installation
============
To install nengo_QIF:

```
git clone https://github.com/Stanford-BIS/nengo_qif
cd nengo_qif
python setup.py develop
```

Usage
=====
  
Here is a minimal example of using this neuron model within Nengo

```
import nengo
import nengo_qif

model = nengo.Network()
with model:
    stim = nengo.Node(lambda t: 0 if t < 0.1 else 1)
    ens = nengo.Ensemble(n_neurons=100, dimensions=1, 
                         neuron_type=nengo_qif.QIF(
                             tau_rc=0.02,    # membrane time constant
                             tau_ref=0.002,  # refractory period
                             ))
    nengo.Connection(stim, ens)
    probe = nengo.Probe(ens, synapse=0.05)
    
sim = nengo.Simulator(model)
sim.run(1)

import pylab
pylab.plot(sim.trange(), sim.data[probe])
pylab.show()
```
