"""
Minimal firetasks for running lammps with pymatlammps.
"""

import os
import json
from fireworks import FiretaskBase, FWAction, explicit_serialize
from fireworks.utilities.fw_serializers import DATETIME_HANDLER
from atomate.utils.utils import get_logger
from atomate.common.firetasks.glue_tasks import get_calc_loc
from atomate.utils.utils import env_chk

from pymatlammps import PyMatLammps
from pymatlammps.fireworks.drone import PMLDrone
from pymatlammps.fireworks.database import PMLCalcDb

logger = get_logger(__name__)


@explicit_serialize
class RunStructurePML(FiretaskBase):
    """Run a series of PyMatLammps/PyLammps methods on an input structure.

    Required params:
        structure (Structure):
            pymatgen structure used to setup lammps simulations domain
        potential_params (dict or list of dicts):
            dictionary specifying the type of lammps potential to be used, and
            any additional parameters to set the specific potential.
            At a minimum the following needs to be included:
            {'type': style of lammps potential (ie pair, bond, angle, etc
             'coeffs': input to corresponding pymatlammps setup function}
             Currently only pair potentials supported. And no modify commands
             but all easily extendable in PyMatLammps class if needed.
        pml_methods (list of tuples)
            list of tuples with each each tuple:
            ('method as string', list of args, dict of kwargs)

    Optional params:
        init_kwargs (dict):
            Keyword arguments used for the initialization of the PyLammps
            interface. See documentation:
            https://lammps.sandia.gov/doc/Python_module.html#the-pylammps-class-api
        additional_setup_commands (list):
            list of lammps commands as str to be run before setting up the
            simulation domain (ie region, box, atoms)
    """

    required_params = ['structure', 'potential_params', 'pml_methods']
    optional_params = ['init_kwargs', 'additional_setup_commands']

    def run_task(self, fw_spec):
        """Setup, run and write output for a structure optimization."""
        pml = PyMatLammps(**self.get('init_kwargs'))
        pml.lmp.commands_list(self.get('additional_setup_commands', []))
        pml.set_structure(self['structure'])

        dump_patterns = ['*.dump*']
        for method in self['pml_methods']:
            getattr(pml, method[0])(*method[1], **method[2])
            if 'dump' in method[0]:
                dump_patterns.append(method[1][0])
            elif 'dump_file' in method[2].keys():
                dump_patterns.append(method[1][2]['dump_file'])

        final_energy = pml.get_potential_energy()
        final_structure = pml.get_structure()
        pml.close()
        logger.info("PyMatLammps finished running.")

        inputs = {
            'structure': self['structure'],
            'potential_params': self['potential_params'],
            'pml_methods': self['pml_methods'],
            'pylammps_init_kwargs': self.get('init_kwargs'),
            'pylammps_setup_commands': self.get('additional_setup_commands')
        }

        outputs = {
            'structure': final_structure,
            'energy': final_energy
        }
        # only return if this is going to be set into the parse task,
        return FWAction(update_spec={'inputs': inputs, 'outputs': outputs,
                                     'dump_file_patterns': dump_patterns})


@explicit_serialize
class PMLtoDB(FiretaskBase):
    """
    Enter a pymatlammps calculations to a database

    Required params:
        inputs (dict):
            dictionary of inputs as passed to RunStructurePML
        outputs (dict):
            dictionary of outputs
            {'structure': final structure, 'energy': final energy}

    optional params:
        calc_dir (str):
            path to dir (on current filesystem) that contains LAMMPS
            output files. Default: use current working directory.
        calc_loc (str OR bool): if True will set most recent calc_loc. If str
            search for the most recent calc_loc with the matching name
        db_file (str): path to file containing the database credentials.
            Supports env_chk. Default: write data to JSON file.
        dump_file_patterns (list):
            list of str for dump file name patterns used to glob dump files.
        additional_fields (dict):
            dict of additional fields to add
    """
    required_params = ['inputs', 'outputs']
    optimal_params = ['calc_dir', 'calc_loc', 'db_file', 'dump_file_patterns',
                      'additional_fields']

    def run_task(self, fw_spec):
        calc_dir = os.getcwd()
        if "calc_dir" in self:
            calc_dir = self["calc_dir"]
        elif self.get("calc_loc"):
            calc_dir = get_calc_loc(self["calc_loc"],
                                    fw_spec["calc_locs"])["path"]

        logger.info("PARSING DIRECTORY: {}".format(calc_dir))

        # find log name
        log_name = 'log.lammps'
        for cmd in self.get('pylammps_setup_commands'):
            if 'log' in cmd:
                log_name = cmd.split(' ')[1]

        # find dump files

        drone = PMLDrone(inputs=self['inputs'],
                         outputs=self['outputs'],
                         log_name=log_name,
                         dump_patterns=self.get('dump_file_patterns', []))

        task_doc = drone.assimilate(calc_dir)

        db_file = env_chk(self.get('db_file'), fw_spec)

        # db insertion
        if not db_file:
            with open('task.json', 'w') as fp:
                json.dump(task_doc, fp, default=DATETIME_HANDLER)
        else:
            mmdb = PMLCalcDb.from_db_file(db_file)
            # insert the task document
            t_id = mmdb.insert(task_doc)
            logger.info(f"Finished parsing with task_id: {t_id}")

        return FWAction(stored_data={"task_id": task_doc.get("task_id", None)})
