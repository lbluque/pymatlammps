"""Simple class to interface PyLammps and pymatgen"""

__author__ = "Luis Barroso-Luque"

import warnings
from ast import literal_eval
from collections import namedtuple
import numpy as np
from scipy.optimize import minimize_scalar
from monty.io import zopen
from lammps import PyLammps
from pymatgen.core.periodic_table import get_el_sp
from pymatgen import SymmOp, Structure, Lattice

SITE_TOL = 1E-12  # fractional tolerance to push atoms into lammps box


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

    def reset(self):
        """Clear all lammps commands and rerun default commands"""
        self.clear()
        self.lmp.commands_list(self.default_cmds)

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
                             algo_params: dict = None, dump_nstep=500,
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

    def optimize_volume(self, box_tol: float = 1E-10, max_iter: int = 5000,
                        algo: str = 'Bounded', algo_params: dict = None,
                        dump_nstep: int = 1,
                        dump_file: str = 'volumeopt.dump'):
        """Optimize lammps box volume using scipy (pymatgen Lattice) only.

        Sites are scaled accordingly such that the optimized volume domain
        is just a scaled version of the input.

        Args:
            tol (float):
                tolerance between changes of box (lattice) between successive
                minimization.
            max_iter (int):
                maximum no. of iterations
            algo (str):
                minimization algorithm.
                See details:
                minimize_scalar from scipy.optimize
            algo_params (dict):
                parameters to set for the minimization algorithm.
                minimize_scalar from scipy.optimize
            dump_nstep (int):
                dump atom positions every nsteps.
            dump_file (str):
                dump file name.
        """
        self.change_box('all', 'triclinic')

        if dump_nstep > 0:
            self.compute('peatom', 'all', 'pe/atom')
            self.dump('vol-relax', 'all', 'custom', dump_nstep, dump_file,
                      'type', 'x', 'y', 'z', 'c_peatom')

        def potential_energy(volume):
            self.scale_box(volume)
            return self.get_potential_energy()

        algo_params = algo_params or {'bounds': (0.001, 20 * self.get_structure().volume)}
        res = minimize_scalar(potential_energy, method=algo,
                              options={'maxiter': max_iter, 'xatol': box_tol},
                              **algo_params)

        if not res['success']:
            raise RuntimeError(f"Minimization did not converge: {res}")

        if dump_nstep > 0:
            self.uncompute('peatom')
            self.undump('vol-relax')

    def scale_box(self, volume: float):
        """Scale the simulation box, using pyamatgen scale lattice

        Args:
            volume (float):
                new volume of unit cell
        """
        structure = self.get_structure()
        structure.scale_lattice(volume)
        lattice = structure.lattice
        xhi, _, _, xy, yhi, _, xz, yz, zhi = lattice.matrix.flatten(order='C')
        self.change_box('all', 'x', 'final', 0.0,  xhi, 'y', 'final', 0.0, yhi,
                        'z', 'final', 0.0, zhi, 'xy', 'final', xy,
                        'xz', 'final', xz, 'yz', 'final', yz, 'remap')

    def set_pair_potential(self, style: list, coeffs: list, mods: list = None,
                           **kwargs):
        """Set a pair style potential for lammps.

        Allows species/elements instead of lammps types to select species.

        Args:
            style (list):
                lammps pair_style given as a list of str.
            coeffs (list):
                lammps pair coefficients as list of lists. Each list can
                include two pymatgen Element/Species included in the system
                instead of the standard lammps input.
            mods (list):
                list of lists allowed pair modify commands.
        """
        coeffs = coeffs if isinstance(coeffs[0], list) else [coeffs]

        if mods is not None and not isinstance(mods[0], list):
            mods = [mods]
        else:
            mods = []

        self.pair_style(*style)

        for coef in coeffs:
            try:
                # lammps only takes sorted atom types...
                atom_types = sorted([self.atom_types[get_el_sp(coef[0])],
                                     self.atom_types[get_el_sp(coef[1])]])
                self.pair_coeff(*atom_types, *coef[2:])
            except ValueError:
                self.pair_coeff(*coef)
        for mod in mods:
            self.pair_modify(*mod)

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
        symmop = self.create_region_from_lattice(
            structure.lattice, 'unit-cell')
        lmp_structure = structure.copy()
        lmp_structure.apply_operation(symmop)
        self.domain = lmp_structure

        if not lmp_structure.lattice.is_orthogonal:
            self.box('tilt', 'large')

        self.create_box(len(structure.composition), 'unit-cell')
        self.create_atoms_from_structure(lmp_structure)

        for sp, i in self.atom_types.items():
            try:
                atomic_mass = sp.atomic_mass
            except AttributeError:
                atomic_mass = 1.0
            self.mass(i, float(atomic_mass))
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
        matrix, rotation = self._get_lower_triangular_matrix(lattice.matrix)

        # TODO try to improve tilt lammps does not like very skewed boxes
        # (borrowed from ase.lammps)
        '''
        flip_order = [(1, 0, 0), (2, 0, 0), (2, 1, 1)]
        flip = np.array([abs(lower[i][j] / lower[k][k]) > 0.5
                         for i, j, k in flip_order])
        for ind, (i, j, k) in enumerate(flip_order):
            if flip[ind]:
                print('flipped')
                change = lower[k][k]
                change *= np.sign(lower[i][j])
                lower[i][j] -= change
        # Then need would need to set new lattice from this and get all the
        # images of site coords that fall outside of the new box
        '''

        xhi, yhi, zhi, xy, xz, yz = matrix[(0, 1, 2, 1, 2, 2),
                                           (0, 1, 2, 0, 0, 1)]
        # assume region origin is always (0, 0, 0)
        # use only one region rn so set id to 1
        self.region(region_name, 'prism', 0, xhi, 0, yhi, 0, zhi, xy, xz, yz)
        symmop = SymmOp.from_rotation_and_translation(rotation)
        return symmop

    @staticmethod
    def _get_lower_triangular_matrix(matrix: np.array) -> np.array:
        """Get lower triangular matrix form of lattice matrix.

        Get a matrix suitable to define a lamps region/box.
        -> lower triangular with all positive diagonal elements

        Try to minimize tilt to make lammps happy as well.

        Args:
            matrix (ndarray):
                A lattice matrix
        Returns:
            ndarry: lower triangular matrix
        """
        rotation, upper = np.linalg.qr(matrix.T, mode='complete')
        # make sure all diagonal elements are positive
        diag = np.array(np.diag(upper) > 0, dtype=int)
        diag[diag == 0] = -1  # mirror negatives
        inversion = np.diag(diag)
        upper = inversion @ upper
        rotation = rotation @ inversion
        return upper.T, rotation.T

    def create_atoms_from_structure(self, structure: Structure):
        """Create lammps atoms from a pymatgen structure

        Args:
            structure (Structure)
        """

        for site in structure:
            # force values near boundary into box to make lammps happy
            #dir = np.argmin(site.frac_coords)
            site.frac_coords[site.frac_coords < 0.0] += 1.0
            site.frac_coords[site.frac_coords > 1.0] -= 1.0
            site.frac_coords += SITE_TOL * (np.array([0.5, 0.5, 0.5]) - site.frac_coords)
            self.create_atoms(self.atom_types[site.specie], 'single',
                              *site.coords, 'units', 'box')

        if len(structure) != len(self.atoms):
            raise RuntimeError(
                f"Only {len(self.atoms)} lammps atoms from {len(structure)} in"
                " the given structure were created.\n"
                "Probably numerical error pushed some coordinates to appear "
                "outside box.\n Check lammps log for more details.")

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
