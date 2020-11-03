# pymatlammps
**Barebones interface between pymatgen and PyLammps API**

...*because the lammps module in pymatgen is too clunky?*

Currently only implemented convenience methods for bulk structures to run
static calculations and energy minimization because that is currently all I
need this for...but if more is needed you get the idea.

So far this is the gist:
```python
from pymatgen import Structure, Lattice
from pymatlammps import PyMatLammps

pml = PyMatLammps()
structure = Structure.from_spacegroup('Fm-3m', Lattice.cubic(2.0),
                                      ['Au'], [[0, 0, 0]])
pml.set_structure(structure)
pml.set_pair_potential(['lj/cut', 4.0], ['Au', 'Au', 1.2, 1.5, 3.0])

energy = pml.get_potential_energy()
pml.optimize_structure(box_tol=1E-12, max_cycles=1000)
optim_energy = pml.get_potential_energy()
relaxed_structure = pml.get_structure()
```

Additionally, some very minimal implementations of classes to launch calcs
using [atomate](https://atomate.org/index.html) + 
[Fireworks](https://materialsproject.github.io/fireworks):

```python
from pymatgen import Structure, Lattice
from fireworks import Workflow, LaunchPad
from pymatlammps.atomate import OptimizeStructureFW

structure = Structure.from_spacegroup('Fm-3m', Lattice.cubic(2.0),
                                      ['Au'], [[0, 0, 0]])

coeffs = [['Au', 'Au', 1.2, 1.5, 3.0]]
pot_params = {'type': 'pair', 'style': ['lj/cut', 3.0], 'coeffs': coeffs}
wf = Workflow([OptimizeStructureFW(structure, pot_params)])

launchpad = LaunchPad.auto_load()
launchpad.add_wf(wf)
```