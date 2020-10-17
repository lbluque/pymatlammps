"""Simple class to interface PyLammps and pymatgen"""

import numpy as np
from numpy.linalg import norm, det, solve
from lammps import PyLammps
from pymatgen import SymmOp
from pymatgen.io.lammps.data import lattice_2_lmpbox


class PyMatLammps(PyLammps):
    """
    Lightly wrap PyLammps to add convenience methods to set up lammps runs from
    pymatgen objects.

    Basically most of the convienience is in setting up the simulation domain
    from a pymatgen object and the reverse, obtaining a pymatgen object from
    a lammps simulation result.
    """

    # default setup commands
    default_cmds = [
        "units metal",
        "atom_style atomic",
        "atom_modify map array sort 0 0.0"
    ]

    def __init__(self, *args, **kwargs):
        if 'cmdargs' not in kwargs.keys():
            kwargs['cmdargs'] = ['-nocite']
        super().__init__(*args, **kwargs)
        self.atom_types = None
        self.lmp.commands_list(self.default_cmds)

    def setup_from_structure(self, structure, sort=True):
        if sort:
            structure.sort()

        self.atom_types = {s: i + 1 for i, s in
                           enumerate(set(structure.species))}

        charge = any(getattr(sp, 'oxi_state', 0) != 0
                     for sp in structure.species)
        if charge:
            self.atom_style('charge')

        # for now assume bulk structures only
        self.boundary('p', 'p', 'p')
        symmop = self._region_from_lattice(structure.lattice, 'unit-cell')
        self.create_box(len(structure), 'unit-cell')
        lmp_structure = structure.copy()
        lmp_structure.apply_operation(symmop)
        self._atoms_from_structure(lmp_structure)

        for sp, i in self.atom_types.items():
            self.mass(i, float(sp.atomic_mass))
            if charge:
                self.set('type', i, 'charge', getattr(sp, 'oxi_state', 0))
        return lmp_structure

    def _region_from_lattice(self, lattice, region_name):
        a, b, c = lattice.abc
        u, v, w = lattice.matrix
        matrix = np.zeros((3, 3), order='C')
        matrix[0, 0] = a  # xhi
        matrix[1, 0] = np.dot(v, u / a)  # xy
        matrix[2, 0] = np.dot(w, u / a)  # xz
        uxv = np.cross(u, v)
        matrix[1, 1] = norm(uxv / a)  # yhi
        matrix[2, 1] = np.dot(w, np.cross(uxv, u / a)) / norm(uxv)  # yz
        matrix[2, 2] = det(lattice.matrix) / norm(uxv)  # zhi
        xhi, _, _, xy, yhi, _, xz, yz, zhi = matrix.flatten()
        # assume region origin is always (0, 0, 0)
        # use only one region rn so set id to 1
        self.region(region_name, 'prism', 0, xhi, 0, yhi, 0, zhi, xy, xz, yz)
        rot = solve(matrix, lattice.matrix)  # rotation matrix
        symmop = SymmOp.from_rotation_and_translation(rot)
        return symmop

    def _atoms_from_structure(self, structure):
        for site in structure:
            self.create_atoms(self.atom_types[site.specie], 'single',
                              *site.coords, 'units', 'box')

    def _lattice_from_structure(self, structure):
        basis = sum((['basis', *crds] for crds in structure.frac_coords), [])
        self.lattice('custom', 1.0,
                     'a1', *structure.lattice.matrix[0],
                     'a2', *structure.lattice.matrix[1],
                     'a3', *structure.lattice.matrix[2],
                     *basis)
