import numpy as np
import pandas as pd

__author__ = 'pslii'


def readASCII(path='rp2.0', filename='orb_elms.dat'):
    """
    Reads 1D ASCII files output by Fortran
    """
    with open(path + '/' + filename, 'r') as fin:
        lines = fin.readlines()

        # parse header
        header = lines[0].split('=')
        variables = [var.strip() for var in header[1].split(',')]
        nvars = len(variables)

        # parse data
        lines = [line.strip() for line in lines[1:]]
        lines = ' '.join(lines).split()

        assert (len(lines) % nvars == 0)
        arr = np.array(lines, dtype=float).reshape((len(lines) / nvars, nvars))
    df = pd.DataFrame(arr, columns=variables)
    return df