import numpy as np


class dataWriter:
    def __init__(self):
        pass

    def writeFortranBinary(self, data, varlist=None):
        """
        Given a dictionary and variable list, 
        writes data to fortran binary file.

        > print(data.keys())
        ['sigma', 'pi']
        > datReader.writeFortranBinary(data, fname='init.dat')

        In fortran:
        real, dimension(nxtot, nytot) :: sigma, pi
        open(13, file='init.dat', form='binary')
        read(13) sigma, pi
        close(13)
        """
        if varlist is None:
            varlist = data.keys()

        from scipy.io import FortranFile

        with FortranFile('init.dat', 'w', '<i') as datfile:
            for var in varlist:
                if not var in data.keys(): continue
                datfile.write_record(np.array(data[var].transpose(),
                                              dtype='<f'))



