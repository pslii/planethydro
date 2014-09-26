from planetHydro.parseData.dataReader import dataReader
from planetHydro.parseData.gridReader import gridReader
from planetHydro.parseData.readParams import readParams

__author__ = 'pslii'


def initialize(path='.'):
    """
    Automatically initializes data reader for current directory.
    Optional inputs: path='.'
    Returns: params, grid, dataReader
    """
    print "Reading parameters..."
    params, _ = readParams(path=path)
    print "Reading grid..."
    grid = gridReader(params, path=path)
    print "Initializing data parser..."
    reader = dataReader(grid, path=path)

    return params, grid, reader