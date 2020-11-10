"""
Minimal CalcDb class to stores objects in database.
Adapted from atomate.
"""

__author__ = "Luis Barroso-Luque"

from pymongo import ASCENDING, DESCENDING
from atomate.utils.database import CalcDb
from atomate.utils.utils import get_logger

logger = get_logger(__name__)


class PMLCalcDb(CalcDb):
    """
    Manage database insertions
    """
    
    def __init__(self, host: str ='localhost', port: int =27017,
                 database: str ='lammps', collection: str ='pmltasks',
                 user: str = None, password: str = None, **kwargs):
        super().__init__(host, port, database, collection, user, password,
                         **kwargs)

    def build_indexes(self, indexes: list = None, background: bool = True):
        """Build database indexes

        Args:
            indexes (list):
                list of single field indexes to be built.
            background (bool):
                Run in the background or not.
        """
        indexes = indexes or []
        self.collection.create_index("task_id", unique=True,
                                     background=background)
        self.collection.create_index([("completed_at", DESCENDING)],
                                     background=background)

        # build single field indices
        for i in indexes:
            self.collection.create_index(i, background=background)

        for formula in ("formula_pretty", "formula_anonymous"):
            self.collection.create_index(
                [
                    (formula, ASCENDING),
                    ("output.energy", DESCENDING),
                ],
                background=background,
            )
            self.collection.create_index(
                [
                    (formula, ASCENDING),
                    ("output.energy_per_atom", DESCENDING),
                ],
                background=background,
            )

    def reset(self):
        self.collection.delete_many({})
        self.db.counter.delete_one({"_id": "taskid"})
        self.db.counter.insert_one({"_id": "taskid", "c": 0})
        self.build_indexes()
