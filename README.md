# pymatlammps
**Barebones interface between pymatgen and PyLammps API**

...because the lammps module in pymatgen is clunky?

Currently only supporting bulk structures and static calculations and
energy minimization because that is all I need this for...but if more is needed
you get the idea.

So far this is the idea:
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