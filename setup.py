from setuptools import setup, find_packages

setup(
    name='PyMatLammps',
    version='0.0.0',
    url='https://github.com/lbluque/pymatlammps',
    author='Luis Barroso-Luque',
    author_email='lbluque@berkeley.edu',
    description='Minimal wrapper to interface ppmlymatgen and PyLammps',
    packages=find_packages(),
    install_requires=['numpy >= 1.18.0', 'pymatgen >= 2020.10.9'],
)
