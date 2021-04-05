"""
Simple Fireworks using pymatlammps
"""

__author__ = "Luis Barroso-Luque"

from fireworks import Firework, FileWriteTask
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from pymatgen import Structure
from pymatlammps.atomate.firetasks import RunStructurePML, PMLtoDB

DB_FILE = None  # '>>db_file<<'


class StaticFW(Firework):
    """Compute the potential energy for a given structure.

    Static calculation, nothing is moved.
    """
    def __init__(self, structure: Structure, potential_params: dict,
                 init_kwargs: dict = None,
                 additional_setup_commands: list = None,
                 parse_dump_files: bool = True,
                 files_to_write: dict = None,
                 name: str = 'PML static calculation',
                 db_file: str = DB_FILE, parents: list = None, **kwargs):
        """
        Args:
            structure (Structure):
                input structure
            potential_params (dict or list of dicts):
                dictionary specifying the style of lammps potential to be used,
                and any additional parameters to set the specific potential.
                At a minimum the following needs to be included:
                {'type': type of lammps potential (ie pair, bond, angle, etc)
                 'style', 'coeffs'}
                 style and coeffs see input to corresponding pymatlammps function
                 Currently only pair potentials supported. And no modify commands
                 but all easily extendable in PyMatLammps class if needed.
            init_kwargs (dict):
                Keyword arguments used for the initialization of the PyLammps
                interface. See documentation:
                https://lammps.sandia.gov/doc/Python_module.html#the-pylammps-class-api
            additional_setup_commands (list):
                list of lammps commands as str to be run before setting up the
                simulation domain (ie region, box, atoms)
            parse_dump_files (bool):
                If true will parse dump files.
            files_to_write (dict):
                dictionary of any files to write using the FileWriteTask.
                Useful to write input files ie for potentials.
            name (str):
                Name for the Firework.
            db_file (str):
                Path to file specifying db credentials to place output parsing.
            parents ([Firework]):
                parent atomate
        """
        tasks = [
            RunStructurePML(
                structure=structure,
                potential_params=potential_params,
                pml_methods=[], init_kwargs=init_kwargs,
                additional_setup_commands=additional_setup_commands
            ),
            PassCalcLocs(name=name),
            PMLtoDB(parse_dump_files=parse_dump_files, db_file=db_file,
                    additional_fields={"task_label": name})
        ]

        if files_to_write is not None:
            tasks.insert(0, FileWriteTask(files_to_write=files_to_write))

        super().__init__(tasks=tasks, parents=parents,
                         name=f"{structure.composition.reduced_formula}-{name}",
                         **kwargs)


class OptimizeStructureFW(Firework):
    """Optimize an input structure for a given potential using lammps.

    This will run lammps minimize with a relax/box fix iteratively.
    See PyMatLammps.optimize_structure for more info.
    """

    def __init__(self, structure: Structure, potential_params: dict,
                 optim_params: dict = None, init_kwargs: dict = None,
                 additional_setup_commands: list = None,
                 parse_dump_files: bool = False,
                 files_to_write: dict = None,
                 name: str = 'PML structure optimization',
                 db_file: str = DB_FILE, parents: list = None, **kwargs):
        """
        Args:
            structure (Structure):
                input structure
            potential_params (dict or list of dicts):
                dictionary specifying the style of lammps potential to be used,
                and any additional parameters to set the specific potential.
                At a minimum the following needs to be included:
                {'type': type of lammps potential (ie pair, bond, angle, etc)
                 'style', 'coeffs'}
                 style and coeffs see input to corresponding pymatlammps function
                 Currently only pair potentials supported. And no modify commands
                 but all easily extendable in PyMatLammps class if needed.
            optim_params (dict):
                keyword arguments for PyMatLammps.optimize_structure, see the
                method doctring for more information.
            init_kwargs (dict):
                Keyword arguments used for the initialization of the PyLammps
                interface. See documentation:
                https://lammps.sandia.gov/doc/Python_module.html#the-pylammps-class-api
            additional_setup_commands (list):
                list of lammps commands as str to be run before setting up the
                simulation domain (ie region, box, atoms)
            parse_dump_files (bool):
                If true will parse dump files.
            files_to_write (dict):
                dictionary of any files to write using the FileWriteTask.
                Useful to write input files ie for potentials.
            name (str):
                Name for the Firework.
            db_file (str):
                Path to file specifying db credentials to place output parsing.
            parents ([Firework]):
                parent atomate
        """
        optim_params = optim_params or {}
        pml_methods = [
            ('optimize_structure', (), optim_params)
        ]

        tasks = [
            RunStructurePML(
                structure=structure,
                potential_params=potential_params,
                pml_methods=pml_methods, init_kwargs=init_kwargs,
                additional_setup_commands=additional_setup_commands
            ),
            PassCalcLocs(name=name),
            PMLtoDB(parse_dump_files=parse_dump_files, db_file=db_file,
                    additional_fields={"task_label": name})
        ]

        if files_to_write is not None:
            tasks.insert(0, FileWriteTask(files_to_write=files_to_write))

        super().__init__(tasks=tasks, parents=parents,
                         name=f"{structure.composition.reduced_formula}-{name}",
                         **kwargs)


class OptimizeVolumeFW(Firework):
    """Optimize volume of input structure for a given potential using lammps.

    This will run lammps minimize with a relax/box fix iteratively with any
    isotropic pressure and forcing structure to allow only scaling.
    See PyMatLammps.optimize_volume for more info.
    """

    def __init__(self, structure: Structure, potential_params: dict,
                 optim_params: dict = None, init_kwargs: dict = None,
                 additional_setup_commands: list = None,
                 parse_dump_files: bool = False,
                 files_to_write: dict = None,
                 name: str = 'PML volume optimization',
                 db_file: str = DB_FILE, parents: list = None, **kwargs):
        """
        Args:
            structure (Structure):
                input structure
            potential_params (dict or list of dicts):
                dictionary specifying the style of lammps potential to be used,
                and any additional parameters to set the specific potential.
                At a minimum the following needs to be included:
                {'type': type of lammps potential (ie pair, bond, angle, etc)
                 'style', 'coeffs'}
                 style and coeffs see input to corresponding pymatlammps function
                 Currently only pair potentials supported. And no modify commands
                 but all easily extendable in PyMatLammps class if needed.
            optim_params
                keyword arguments for PyMatLammps.optimize_volume, see the
                method doctring for more information.
            init_kwargs (dict):
                Keyword arguments used for the initialization of the PyLammps
                interface. See documentation:
                https://lammps.sandia.gov/doc/Python_module.html#the-pylammps-class-api
            additional_setup_commands (list):
                list of lammps commands as str to be run before setting up the
                simulation domain (ie region, box, atoms)
            parse_dump_files (bool):
                If true will parse dump files.
            files_to_write (dict):
                dictionary of any files to write using the FileWriteTask.
                Useful to write input files ie for potentials.
            name (str):
                Name for the Firework.
            db_file (str):
                Path to file specifying db credentials to place output parsing.
            parents ([Firework]):
                parent atomate
        """
        optim_params = optim_params or {}
        pml_methods = [
            ('optimize_volume', (), optim_params)
        ]

        tasks = [
            RunStructurePML(
                structure=structure,
                potential_params=potential_params,
                pml_methods=pml_methods, init_kwargs=init_kwargs,
                additional_setup_commands=additional_setup_commands
            ),
            PassCalcLocs(name=name),
            PMLtoDB(parse_dump_files=parse_dump_files, db_file=db_file,
                    additional_fields={"task_label": name})
        ]

        if files_to_write is not None:
            tasks.insert(0, FileWriteTask(files_to_write=files_to_write))

        super().__init__(tasks=tasks, parents=parents,
                         name=f"{structure.composition.reduced_formula}-{name}",
                         **kwargs)
