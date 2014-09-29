#!/usr/local/lib/miniconda/bin/python
import numpy.f2py as f2py
import numpy as np


class outputTec:
    spc = '      '
    nxt = '     $'

    def __init__(self, varlist, grid, outDim=None,
                 suffix='xy.dat', path='.',
                 output=False):
        """

        :type varlist: list
        :param varlist:
        :type grid: planetHydro.parseData.gridReader.gridReader
        :param grid:
        :type outDim: int
        :param outDim:
        :type suffix: str
        :param suffix:
        :type path: str
        :param path:
        :type output: bool
        :param output:
        """
        self.grid = grid
        self.ndims = grid.ndims if outDim is None else outDim
        if self.ndims == 2:
            self.gridsig = (['x', 'y'], 'nxtot, nytot')
            self.nztot = 1
            self.shape = self.nxtot, self.nytot = grid.nxtot, grid.nytot + 1
        elif self.ndims == 3:
            self.gridsig = (['x', 'y', 'z'], 'nxtot, nytot, nztot')
            self.shape = self.nxtot, self.nytot, self.nztot = grid.nxtot, grid.nytot + 1, grid.nztot
        else:
            assert False, "Error: invalid output dimension {0}".format(outDim)

        self.varlist = varlist
        self.suffix = suffix
        self.varstring = ','.join(varlist)
        self.fortranvarlist = (',\n' + self.nxt + ' ').join(self.gridsig[0] + varlist)
        self.path = path

        self.source = self.genSource()
        if output:
            with open(path + "/tecout.f", "w") as txt_file:
                txt_file.write(self.source)

        try:
            import os
            os.remove(path + '/tecout.so')
        except OSError:
            print "No such file."
            pass

        # paths set for M10
        self.modulename = 'tecout' + str(self.ndims) + 'D'
        f2py.compile(self.source, modulename=self.modulename,
                     extra_args='-I/data/programs/local/src/tecio-wrap/ ' + \
                                '--f90exec=/opt/intel/composer_xe_2011_sp1.8.273/bin/intel64/ifort ' + \
                                '-L/data/programs/local/lib/ -ltecio -lstdc++')

    def bind(self, string):
        nullchr = '//nullchr'
        return '"' + string + '"' + nullchr

    def genHEADER(self):
        spc, nxt = self.spc, self.nxt

        call = ['C File tecoutput.f',
                spc + 'subroutine tecoutput(ndat,' + self.fortranvarlist + ')',
                spc + 'use tecio',
                spc + 'integer, parameter :: nxtot=' + str(self.nxtot) + ', nytot=' + str(
                    self.nytot) + ', nztot=' + str(self.nztot),
                spc + 'integer, intent(in) :: ndat',
                spc + 'character, parameter :: nullchr=char(0)',
                spc + 'character*100 :: fname',
                spc + 'real, intent(in), dimension(' + self.gridsig[1] + ') ::',
                nxt + self.fortranvarlist + '\n\n']
        return '\n'.join(call)

    def genTECINI(self, Title='xy', ScratchDir='.', VIsDouble=0):
        spc, nxt = self.spc, self.nxt
        name = spc + "write(fname,'(i4.4 " + '"' + self.suffix + '"' + ")') ndat\n"
        name += spc + "write(*,*) fname\n"
        Title = self.bind(Title)
        Variables = self.bind(self.fortranvarlist)
        Fname = 'trim(fname)//nullchr'
        ScratchDir = self.bind(ScratchDir)
        Debug, VIsDouble = '0', str(VIsDouble)
        call = (',\n' + nxt + '\t').join([Title, Variables,
                                          Fname, ScratchDir, Debug, VIsDouble])
        return name + (spc + 'call tecini(' + call + ')')

    def genTECZNE(self, zone_title):
        spc, nxt = self.spc, self.nxt

        ZoneTitle = self.bind(zone_title)
        L, M, N = str(self.nxtot), str(self.nytot), str(self.nztot)
        ZoneFormat = self.bind('BLOCK')
        DupList = "nullchr"
        call = (',\n' + nxt + '\t').join([ZoneTitle, L, M, N,
                                          ZoneFormat, DupList])
        return spc + 'call teczne(' + call + ')'

    def genTECGRID(self, IsDouble=0):
        call = self.genTECDAT('x', IsDouble=IsDouble) + '\n' + \
               self.genTECDAT('y', IsDouble=IsDouble)
        if self.ndims == 3:
            call += '\n' + self.genTECDAT('z', IsDouble=IsDouble)
        return call

    def genTECDAT(self, VarName, IsDouble=0):
        call = ', '.join(['nxtot*nytot*nztot', VarName, str(IsDouble)])
        return self.spc + 'call tecdat(' + call + ')'

    def genTECEND(self):
        return self.spc + 'call tecend()'

    def genFOOTER(self):
        return self.spc + 'end subroutine tecoutput \nC end File tecoutput.f'

    def genSource(self):
        calls = [self.genHEADER(), self.genTECINI(), self.genTECZNE('xy'), self.genTECGRID()]

        for var in self.varlist:
            calls.append(self.genTECDAT(var))
        calls.append(self.genTECEND())
        calls.append(self.genFOOTER())
        return '\n'.join(calls)

    def extend(self, arr):
        if self.ndims == 2:
            last = arr[:, -1]
            return np.append(arr, last[:, np.newaxis], 1)
        elif self.ndims == 3:
            last = arr[:, -1, :]
            return np.append(arr, last[:, np.newaxis, :], 1)

    def writeNDat(self, ndat, data, phi):
        _x, _y, _z = self.grid.plotCoords(phi, ndims=self.ndims)

        import sys, os
        sys.path.append(os.getcwd())
        tecout = __import__(self.modulename)
        reload(tecout)

        for _var in self.varlist:
            assert (_var in data.keys()), "Error: {0} not found in data".format(_var)
            _temp = self.extend(data.get(_var))
            assert _temp.shape == self.shape, "Error: shape mismatch {0}, {1}".format(_temp.shape, self.shape)
            exec (_var + ' = _temp')

        if self.ndims == 2:
            gridvar = '_x,_y'
        elif self.ndims == 3:
            gridvar = '_x,_y,_z'
        else:
            assert False

        exec ('tecout.tecoutput(' + str(ndat) + ',' + gridvar + ',' +
              self.varstring + ')')


def detectData(path='.', n_start=0):
    from os.path import isfile
    # Detect number of files to be processed
    """
    start = n_start
    while (isfile(path+'/'+str(start).zfill(4)+'xy.dat')):
        start += 1
    """
    start = 0
    end = start
    while (isfile(path + '/' + str(end).zfill(4) + 'dat') and
               not isfile(path + '/' + str(end).zfill(4) + '.xy.dat')):
        end += 1
    return start, end, end - start