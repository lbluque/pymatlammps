"""Simple class to interface PyLammps and pymatgen"""

import warnings
from ast import literal_eval
from collections import namedtuple
import numpy as np
from numpy.linalg import norm, det, solve
from monty.io import zopen
from lammps import PyLammps
from pymatgen.core.periodic_table import get_el_sp
from pymatgen import SymmOp, Structure, Lattice, Species, Element


__author__ = "Luis Barroso-Luque"


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
        "neigh_modify every 1 delay 5 check yes"
    ]

    def __init__(self, *args, **kwargs):
        if 'cmdargs' not in kwargs.keys():
            kwargs['cmdargs'] = ['-nocite']
        super().__init__(*args, **kwargs)
        self.atom_types = None
        self.domain = None
        self.lmp.commands_list(self.default_cmds)

    @property
    def species_types(self):
        """Return the inverse mapping of atom_types"""
        if self.atom_types is not None:
            return {v: k for k, v in self.atom_types.items()}

    def get_potential_energy(self) -> float:
        """Get the potential energy evaluated by lammps."""
        _ = self.run(0)  # force evaluation
        return self.variables['pe'].value

    def get_structure(self) -> Structure:
        """Get a Structure for current lammps state"""
        lattice = self.get_lattice()
        species = [self.species_types[self.atoms[i].type]
                   for i in range(len(self.atoms))]
        coords = [self.atoms[i].position for i in range(len(self.atoms))]
        return Structure(lattice, species, coords, coords_are_cartesian=True)

    def get_lattice(self) -> Lattice:
        """Get a Lattice for current lammps state"""
        lo, hi, xy, yz, xz, _, _ = self.lmp.extract_box()
        bounds = np.array(hi) - np.array(lo)
        matrix = [[bounds[0], 0, 0], [xy, bounds[1], 0], [xz, yz, bounds[2]]]
        return Lattice(matrix)

    def optimize_site_coords(self, energy_tol=0.0, force_tol=1E-10,
                             max_iter=5000, max_eval=5000, algo='cg',
                             algo_params: dict = None, dump_nstep=1,
                             dump_file='coordopt.dump'):
        """Optimize structure site coordinates using lammps minimize.

        This does not use a fix to allow cell relaxation. For full relaxation
        use the optimize_structure method.

        Args:
            energy_tol (float):
                energy stopping tolerance
            force_tol (float):
                force stopping tolerance
            max_iter (int):
                maximum no. of iterations
            max_eval (int):
                maximum number of force evaluations
            algo (str):
                minimization algorithm. See for details:
                https://lammps.sandia.gov/doc/min_style.html
            algo_params (dict):
                parameters to set for the minimization algorithm.
            dump_nstep (int):
                dump atom positions every nsteps.
            dump_file (str):
                dump file name.
        """
        self.reset_timestep(0)
        if dump_nstep > 0:
            self.compute('peatom', 'all', 'pe/atom')
            self.dump('coord-relax', 'all', 'custom', dump_nstep, dump_file,
                      'type', 'x', 'y', 'z', 'c_peatom')

        self.min_style(algo)
        if algo_params is not None:
            for key, vals in algo_params.items():
                self.min_modify(key, vals)
        _ = self.minimize(energy_tol, force_tol, max_iter, max_eval)

        if dump_nstep > 0:
            self.uncompute('peatom')
            self.undump('coord-relax')

    def optimize_structure(self, box_tol: float = 1E-8,
                           energy_tol: float = 0.0, force_tol: float = 1E-10,
                           max_iter: int = 1000, max_eval: int = 1000,
                           max_cycles: int = 100, algo: str = 'cg',
                           algo_params: dict = None,
                           box_fix_params: dict = None, dump_nstep: int = 1,
                           dump_file: str ='structopt.dump'):
        """Perform a structure optimization using lammps minimize.

        This will perform a series of cycles restarting the minimization for
        the latest box parameters, this allows lamps to reset the minimization
        objective function to the current box dimensions and improve the
        obtained local minima.

        A target stress will be set using the box/relax fix (default to zero
        stress) to optimize the box shape.

        akin to:
        https://www.ctcms.nist.gov/potentials/iprPy/notebook/relax_static.html

        Args:
            box_tol (float):
                tolerance between changes of box (lattice) between successive
                minimizations
            energy_tol (float):
                energy stopping tolerance
            force_tol (float):
                force stopping tolerance
            max_iter (int):
                maximum no. of iterations
            max_eval (int):
                maximum number of force evaluations
            max_cycles (int):
                maximum cycles to reset and run minimization
            algo (str):
                minimization algorithm. See for details:
                https://lammps.sandia.gov/doc/min_style.html
            algo_params (dict):
                parameters to set for the minimization algorithm.
            box_fix_params (dict):
                parameters defining a lammps box/relax fix, excluding the ID
                The default will allow 6 box parameters to change independently
                at zero stress (tri 0.0)
                https://lammps.sandia.gov/doc/fix_box_relax.html
            dump_nstep (int):
                dump atom positions every nsteps.
            dump_file (str):
                dump file name.
        """
        if not isinstance(self.domain, Structure):
            warnings.warn(f"Lammps was setup using a {type(self.domain)}"
                          f" and not a {Structure}.\n Make sure that is what "
                          "you want.")

        prev_matrix = self.get_lattice().matrix

        self.change_box('all', 'triclinic')
        if box_fix_params is None:
            self.fix('relax', 'all', 'box/relax', 'tri', 0.0,
                     'fixedpoint', 0, 0, 0)
        else:
            self.fix('relax', 'all', 'box/relax', 'fixedpoint', 0, 0, 0,
                     *(i for item in box_fix_params.items() for i in item))

        converged = False
        for i in range(max_cycles):
            self.optimize_site_coords(energy_tol, force_tol, max_iter,
                                      max_eval, algo, algo_params,
                                      dump_nstep, f'{dump_file}.{i}')
            if np.allclose(self.get_lattice().matrix, prev_matrix,
                           rtol=box_tol):
                converged = True
                break
            prev_matrix = self.get_lattice().matrix

        self.unfix('relax')

        if not converged:
            raise RuntimeWarning("Structure optimization failed to converge "
                                 "to the given tolerances.")

    def optimize_volume(self, box_tol: float = 1E-8,
                        energy_tol: float = 0.0, force_tol: float = 1E-10,
                        max_iter: int = 1000, max_eval: int = 1000,
                        max_cycles: int = 100, algo: str = 'cg',
                        algo_params: dict = None, dump_nstep: int = 1,
                        dump_file: str = 'volumeopt.dump'):
        """Optimize lammps box volume (pymatgen Lattice) only.

        Sites are scaled accordingly such that the optimized volume domain
        is just a scaled version of the input.

        To do this simply run the optimize structure command allowing box
        relaxations and reset sites to original locations, scaled accordingly.

        Args:
            box_tol (float):
                tolerance between changes of box (lattice) between successive
                minimization.
            energy_tol (float):
                energy stopping tolerance
            force_tol (float):
                force stopping tolerance
            max_iter (int):
                maximum no. of iterations
            max_eval (int):
                maximum number of force evaluations
            max_cycles (int):
                maximum cycles to reset and run minimization
            algo (str):
                minimization algorithm. See for details:
                https://lammps.sandia.gov/doc/min_style.html
            algo_params (dict):
                parameters to set for the minimization algorithm.
                https://lammps.sandia.gov/doc/fix_box_relax.html
            dump_nstep (int):
                dump atom positions every nsteps.
            dump_file (str):
                dump file name.
        """
        self.optimize_structure(box_tol, energy_tol, force_tol, max_iter,
                                max_eval, max_cycles, algo, algo_params,
                                box_fix_params={'iso': 0.0},
                                dump_nstep=dump_nstep, dump_file=dump_file)
        vol_opt = self.get_structure().volume
        structure = self.domain.copy()
        structure.scale_lattice(vol_opt)

        # update lammps atom positions
        for i, coords in enumerate(structure.cart_coords):
            self.atoms[i].position = coords

    def set_pair_potential(self, style: list, *coeffs: list):
        """Set a pair style potential for lammps.

        Allows species/elements instead of lammps types to select species.

        Args:
            style (list):
                lammps pair_style given as a list of str.
            coeffs (list):
                lammps pair coefficients as list of lists. Each list can
                include two pymatgen Element/Species included in the system
                instead of the standard lammps input.
        """
        self.pair_style(*style)
        for coef in coeffs:
            try:
                # lammps only takes sorted atom types...
                atom_types = sorted([self.atom_types[get_el_sp(coef[0])],
                                     self.atom_types[get_el_sp(coef[1])]])
                self.pair_coeff(*atom_types, *coef[2:])
            except ValueError:
                self.pair_coeff(*coeffs)

    def set_structure(self, structure: Structure, sort=True):
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

    def create_region_from_lattice(self, lattice: Lattice,
                                   region_name: str) -> SymmOp:
        """Set a lammps region from a pymatgen Lattice object

        Args:
            lattice (Lattice):
                Lattice to use to define the lammps region.
            region_name (str):
                name to use as lammps ID for the region.
        Returns:
            SymmOp: symmetry operations necessary to convert to lammps
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

    def create_atoms_from_structure(self, structure: Structure):
        """Create lammps atoms from a pymatgen structure

        Args:
            structure (Structure)
        """
        for site in structure:
            site.coords += 1E-14  # force small negatives to positive
            self.create_atoms(self.atom_types[site.specie], 'single',
                              *site.coords, 'units', 'box')

        if len(structure) != len(self.atoms):
            raise RuntimeError(f"Only {len(self.atoms)} lammps atoms from "
                               f"{len(structure)} in the given structure "
                               "were created. \nProbably numerical error made "
                               "coordinates appear outside box.\n"
                               "Check lammps log for more details.")

    def create_lattice_from_structure(self, structure: Structure):
        basis = sum((['basis', *crds] for crds in structure.frac_coords), [])
        self.lattice('custom', 1.0,
                     'a1', *structure.lattice.matrix[0],
                     'a2', *structure.lattice.matrix[1],
                     'a3', *structure.lattice.matrix[2],
                     *basis)

    def get_dump_trajectory(self, file: str):
        """Get pyamatgen structures and corresponding properties.

        Properties computed and saved to a dump file by lammps.

        Args:
            file (str):
                file path to lammps dump file

        Returns:
            list of named tuple: trajectory with structure and properties
        """
        dump = parse_dump(file)

        Entry = namedtuple('Entry', ['timestep', 'structure'])
        trajectory = []
        for entry in dump:
            species, coords, properties = [], [], []
            for item in entry['data']:
                species.append(self.species_types[item['type']])
                coords.append((item['x'], item['y'], item['z']))
                properties.append({k: v for k, v in item.items()
                                   if k not in ('x', 'y', 'z', 'type')})

            structure = Structure(Lattice.from_dict(entry['lattice']),
                                  species=species, coords=coords,
                                  coords_are_cartesian=True)
            for site, props in zip(structure, properties):
                site.properties.update(props)
            trajectory.append(Entry(entry['timestep'], structure))

        return trajectory


def parse_dump(file: str):
    """Parse a lammps dump file

    Args:
        file (str):
            file path to lammps dump file

    Returns:
        dict: Dictionary with data in dump
    """
    entries = []
    with zopen(file, 'r') as fp:
        cache = []
        for line in fp:
            if line.startswith("ITEM: TIMESTEP"):
                if len(cache) > 0:
                    entry = {'timestep': int(cache[1]),
                             'natoms': int(cache[3]), 'data': []}
                    matrix = np.zeros((3, 3))
                    bounds = np.array([[float(i) for i in row.split(' ')]
                                       for row in cache[5:8]])
                    for i, bound in enumerate(bounds):
                        matrix[i, i] = bound[0] - bound[1]
                    if bounds.shape == (3, 3):  # tilts
                        matrix[1, 0] = bounds[0, 2]
                        matrix[2, 0] = bounds[1, 2]
                        matrix[2, 1] = bounds[2, 2]
                    entry['lattice'] = Lattice(matrix).as_dict()
                    keys = cache[8].replace("ITEM: ATOMS", "").split()
                    entry['data'] = [
                        {key: literal_eval(value) for key, value
                         in zip(keys, values.split(' '))}
                        for values in cache[9:]]
                    entries.append(entry)
                cache = [line]
            else:
                cache.append(line)
    return entries
