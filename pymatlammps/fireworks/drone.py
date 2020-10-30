import os
from datetime import datetime
from monty.io import zopen
from monty.json import jsanitize
from pymatgen import Composition, Structure
from pymatgen.apps.borg.hive import AbstractDrone
from pymatlammps.core import parse_dump
from pymatlammps import __version__ as pml_version

from atomate.utils.utils import get_logger
logger = get_logger(__name__)


class PMLDrone(AbstractDrone):
    """Drone to parse through results computed with PyMatLammps"""

    __version__ = pml_version

    schema = {
        "root": {
            "schema", "dir_name", "chemsys", "composition_reduced",
            "formula_pretty", "formula_reduced_abc", "elements",
            "nelements", "formula_anonymous", "completed_at",
            "input", "output", "state"
        },
        "input": {'structure', 'potential_params', 'pml_methods',
                  'pylammps_init_kwargs', 'additional_commands'},
        "output": {'structure', 'energy', 'energy_per_atom', 'density',
                   'log', 'dumps'},
    }

    def __init__(self, inputs: dict = None, outputs: dict = None,
                 log_name: str ='log.lammps', dump_names: list = None):
        """Initialize drone

        Args:
            inputs (dict):
                dictionary with input to pymatlammps with schema specified
                above.
            outputs (dict)
                dictionary with output from pymatlammps with schema specified
                above.
            log_name (str):
                file name of lammps log
            dump_names (list of str):
                list of names of dump files written by lammps
        """
        self.input = inputs
        self.output = outputs
        self.log_name = log_name
        self.dump_names = dump_names or []

    def assimilate(self, path: str) -> dict:
        """Assimilate output from a pymatlammps run.

        Basically just copy log file and parse dump files.
        """

        log = {'path': os.path.join(path, self.log_name)}
        with zopen(log['path'], 'r') as fp:
            log['contents'] = [line for line in fp.readlines()]

        dumps = [parse_dump(os.path.join(path, fname))
                 for fname in self.dump_names]
        doc = self.generate_doc(path, log, dumps)

        # maybe check schema is good?
        return doc

    def generate_doc(self, dir_name: str, log: dict, dumps: list) -> dict:
        """Generate doc for db insertion

        Args:
            dir_name (str):
                path to run directory
            log (dict):
                dict with path and list of all read lines in lammps log file.
            dumps (list):
                list of parse lammps dump files.
        Returns:
            dict
        """
        doc = {}
        try:
            doc['schema'] = {'code': 'pymatlammps',
                             'version': PMLDrone.__version__}
            doc['completed_at'] = str(datetime.fromtimestamp(os.path.getmtime(log['path'])))
            doc['dir_name'] = os.path.abspath(dir_name)
            doc['input'] = jsanitize(self.input)
            doc['output'] = jsanitize(self.output)
            doc['output']['log'] = log
            doc['output']['dumps'] = jsanitize(dumps)
            final_structure = self.output['structure']
            composition = final_structure.composition
            doc['output']['energy_per_atom'] = doc['output']['energy']/len(final_structure)
            doc['output']['density'] = final_structure.density
            doc['composition_reduced'] = Composition(composition.reduced_formula).as_dict()
            doc['formula_pretty'] = composition.reduced_formula
            doc['formula_reduced_abc'] = composition.reduced_composition.alphabetical_formula
            doc['formula_anonymous'] = composition.anonymized_formula
            doc['elements'] = [str(e) for e in composition.elements]
            doc['nelements'] = len(doc['elements'])
            doc['chemsys'] = "-".join(sorted(doc["elements"]))
        except:
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"Error in {os.path.abspath(dir_name)}.\n"
                         f"{traceback.format_exc()}")
            return None

        return doc

    def get_valid_paths(self, path: str):  # what's the point in this case?
        return [path]

    def as_dict(self) -> dict:
        d = {'input': jsanitize(self.input),
             'output': jsanitize(self.output),
             'log_name': self.log_name,
             'dump_names': self.dump_names}
        return d

    @classmethod
    def from_dict(cls, d: dict):
        d['input']['structure'] = Structure.from_dict(d['input']['structure'])
        d['output']['structure'] = Structure.from_dict(d['input']['structure'])
        return cls(**d)
