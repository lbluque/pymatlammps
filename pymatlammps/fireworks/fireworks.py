"""
Simple Fireworks using pymatlammps
"""

from fireworks import Firework
from atomate.common.firetasks.glue_tasks import PassCalcLocs
from pymatgen import Structure
from pymatlammps.fireworks.firetasks import RunStructurePML, PMLtoDB


class OptimizeStructureFW(Firework):
    """Optimize an input structure for a given potential using lammps.

    This will run lammps minimize with a relax/box fix iteratively.
    See PyMatLammps.optimize_structure for more info.
    """

    def __init__(self, structure: Structure, potential_params: dict,
                 optim_params: dict = None, init_kwargs: dict = None,
                 additional_setup_commands: list = None,
                 name: str = 'PML structure optimization',
                 db_file: str = None, parents: list = None, **kwargs):
        """
        Args:
            structure (Structure):
                input structure
            potential_params (dict):
                dictionary specifying the type of lammps potential to be used,
                and any additional parameters to set the specific potential.
                At a minimum the following needs to be included:
                {'type': style of lammps potential (ie pair, bond, angle, etc
                 'coeffs': input to corresponding pymatlammps setup function}
                 Currently only pair potentials supported. And no modify
                 commands but all easily extendable in PyMatLammps class when
                 needed.
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
            name (str):
                Name for the Firework.
            db_file (str):
                Path to file specifying db credentials to place output parsing.
            parents ([Firework]):
                parent fireworks
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
            PMLtoDB(db_file=db_file, additional_fields={"task_label": name})
        ]
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
                 name: str = 'PML structure optimization',
                 db_file: str = None, parents: list = None, **kwargs):
        """
        Args:
            structure (Structure):
                input structure
            potential_params (dict):
                dictionary specifying the type of lammps potential to be used,
                and any additional parameters to set the specific potential.
                At a minimum the following needs to be included:
                {'type': style of lammps potential (ie pair, bond, angle, etc
                 'coeffs': input to corresponding pymatlammps setup function}
                 Currently only pair potentials supported. And no modify
                 commands but all easily extendable in PyMatLammps class when
                 needed.
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
            name (str):
                Name for the Firework.
            db_file (str):
                Path to file specifying db credentials to place output parsing.
            parents ([Firework]):
                parent fireworks
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
            PMLtoDB(db_file=db_file, additional_fields={"task_label": name})
        ]
        super().__init__(tasks=tasks, parents=parents,
                         name=f"{structure.composition.reduced_formula}-{name}",
                         **kwargs)
