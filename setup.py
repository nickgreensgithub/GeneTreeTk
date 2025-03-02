from setuptools import setup

import os

def version():
    setupDir = os.path.dirname(os.path.realpath(__file__))
    versionFile = open(os.path.join(setupDir, 'genetreetk', 'VERSION'))
    return versionFile.readline().strip()

setup(
    name='GeneTreeTk',
    version=version(),
    author='Donovan Parks',
    author_email='donovan.parks@gmail.com',
    packages=['genetreetk'],
    scripts=['bin/genetreetk'],
    package_data={'genetreetk' : ['VERSION']},
    url='http://pypi.python.org/pypi/genetreetk/',
    license='GPL3',
    description='A toolbox for working with gene trees.',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy >= 1.8.0",
        "biolib @ git+https://git@github.com/nickgreensgithub/biolib.git",
        "dendropy >= 4.0.0",
        "extern"
        ],
)
