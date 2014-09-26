import re
import string

from planetHydro.parseData.dataReader import _sandbox


__author__ = 'pslii'


def readParams(verbose=False, path='.'):
    """
    Parses param.inc files. For best results, use Fortran90 formatting in
    param.inc.
    """
    evallist = []

    with open(path + '/param.inc', 'r') as paramfile:
        params = paramfile.readlines()
        regex_flt = re.compile(r'([-+]?[(\d*\.?\d+)|(\d+\.?\d*)])[eEdD]([-+]?\d+)')
        regex_f = re.compile(r'\.false\.')
        regex_t = re.compile(r'\.true\.')
        params = [regex_f.sub('False', line) for line in params]
        params = [regex_t.sub('True', line) for line in params]
        for line in params:
            # handle full line comments, empty strings, lines w/ nothing defined
            if (line.isspace() or
                    (not ('=' in line)) or
                    (line[0] != ' ')): continue

            # handle partial line comments
            if '!' in line:
                line = line.split('!')[0]

            # strip first 6 chars
            line = line[6:].strip()

            # strip out param definition
            if '::' in line:
                split = line.split('::')
                assert (len(split) == 2)
                line = split[1]
            if 'parameter(' in line:
                line = line.split('parameter(')[1]

            # parse variables
            line = line.strip(string.whitespace + ',').split(',')

            for var in line:
                var = var.strip()
                if var == '': continue

                # handle unmatched parentheses
                if var.count('(') < var.count(')'):
                    var = var.rstrip(')')

                var = regex_flt.sub(r'\1e\2', var)
                evallist.append(var)

    params_dict, varlist = _sandbox(evallist)
    if verbose: print 'Parameters loaded:\n', string.join(varlist)
    return params_dict, varlist