"""Simple class to interface PyLammps and pymatgen"""

import numpy as np
from numpy.linalg import norm, det, solve
from lammps import PyLammps
from pymatgen import SymmOp


class PyMatLammps(PyLammps):
    """
    Lightly wrap PyLammps to add convenience methods to set up lammps runs from
    pymatgen objects.

    Basically most of the convienience is in setting up the simulation domain
    from a pymatgen object and the reverse, obtaining a pymatgen object from
    a lammps simulation result.

    Attributes:
        atom_types (dict):
            dictionary of IDs for different atom types. Keys are pymatgen
            Element/Species, values are the lammps ids.
        domain (Structure):
            pymatgen object representing initial lammps simulation domain.
            Including the applied operations necessary for lammps input.
            Currently only structures.
    """

    # default setup commands
    default_cmds = [
        "units metal",
        "atom_style atomic",
        "atom_modify map array sort 0 0.0",
        "variable pxx equal pxx",
        "variable pyy equal pyy",
        "variable pzz equal pzz",
        "variable pxy equal pxy",
        "variable pxz equal pxz",
        "variable pyz equal pyz",
        "variable fx atom fx",
        "variable fy atom fy",
        "variable fz atom fz",
        "variable pe equal pe",
        "neigh_modify every 1 delay 0 check yes"
    ]

    def __init__(self, *args, **kwargs):
        if 'cmdargs' not in kwargs.keys():
            kwargs['cmdargs'] = ['-nocite']
        super().__init__(*args, **kwargs)
        self.atom_types = None
        self.domain = None
        self.lmp.commands_list(self.default_cmds)

    def set_structure(self, structure, sort=True):
        """Setup a lammps simulation domain from a pymatgen structure

        Currently only bulk 3D structure, since thats all I'm using this for.

        Args:
            structure (Structure):
                Structure object to use to setup Lammps simulation. Must be
                an ordered structure.
            sort (bool):
                Sort the sites in the structure.
        """
        if sort:
            structure.sort()

        self.atom_types = {s: i + 1 for i, s in
                           enumerate(structure.composition)}

        charge = any(getattr(sp, 'oxi_state', 0) != 0
                     for sp in structure.species)
        if charge:
            self.atom_style('charge')

        # for now assume bulk structures only
        self.boundary('p', 'p', 'p')
        symmop = self.create_region_from_lattice(structure.lattice, 'unit-cell')
        lmp_structure = structure.copy()
        lmp_structure.apply_operation(symmop)
        self.domain = lmp_structure

        if not lmp_structure.lattice.is_orthogonal:
            self.box('tilt', 'large')

        self.create_box(len(structure.composition), 'unit-cell')
        self.create_atoms_from_structure(lmp_structure)

        for sp, i in self.atom_types.items():
            self.mass(i, float(sp.atomic_mass))
            if charge:
                self.set('type', i, 'charge', getattr(sp, 'oxi_state', 0))

    def create_region_from_lattice(self, lattice, region_name):
        """Set a lammps region from a pymatgen Lattice object

        Args:
            lattice (Lattice):
                Lattice to use to define the lammps region.
            region_name (str):
                name to use as lammps ID for the region.
        """
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
        xhi, _, _, xy, yhi, _, xz, yz, zhi = matrix.flatten(order='C')
        # assume region origin is always (0, 0, 0)
        # use only one region rn so set id to 1
        self.region(region_name, 'prism', 0, xhi, 0, yhi, 0, zhi, xy, xz, yz)
        rotation_matrix = solve(matrix, lattice.matrix)
        symmop = SymmOp.from_rotation_and_translation(rotation_matrix)
        return symmop

    def create_atoms_from_structure(self, structure):
        """Create lammps atoms from a pymatgen structure

        Args:
            structure (Structure)
        """
        for site in structure:
            # make small enough cords exactly zero so lammps is happy
            site.coords += 1E-15  # force small negatives to positive
            self.create_atoms(self.atom_types[site.specie], 'single',
                              *site.coords, 'units', 'box')

        if len(structure) != len(self.atoms):
            raise RuntimeError(f"Only {len(self.atoms)} lammps atoms from "
                               f"{len(structure)} in the given structure "
                               "were created. \nProbably numerical error made "
                               "coordinates appear outside box.\n"
                               "Check lammps log for more details.")

    def create_lattice_from_structure(self, structure):
        basis = sum((['basis', *crds] for crds in structure.frac_coords), [])
        self.lattice('custom', 1.0,
                     'a1', *structure.lattice.matrix[0],
                     'a2', *structure.lattice.matrix[1],
                     'a3', *structure.lattice.matrix[2],
                     *basis)

    # TODO easy way to setup force fields
    # TODO easy way to set up minimization command
    # TODO easy way to run minimization and export pymatgen structures and
    #  corresponding energy...then we are ready to go into some atomate perhaps
