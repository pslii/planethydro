__author__ = 'pslii'
import numpy as np
import pandas as pd

def fix_ascii_file(path='.', filename="orb_elms.dat"):
    """
    Repairs ASCII data output file which is missing data.
    :param path:
    :param filename:
    :return:
    """
    with open(path+'/'+filename, 'r') as fin:
        lines = fin.readlines()

        # parse header
        header = lines[0].split('=')
        variables = [var.strip() for var in header[1].split(',')]
        nvars = len(variables)

        # number of lines per timestep
        lines_per_timestep = int(np.ceil(nvars/3.0))

        lines = [line.strip() for line in lines[1:]]
        i = 0

        output_lines = []
        while (i < len(lines)-lines_per_timestep):
            line = ' '.join(lines[i:i+lines_per_timestep]).split()
            if len(line) == nvars:
                output_lines.append(line)
                i += lines_per_timestep
            elif len(line) > nvars: # missing line break
                output_lines.append(line[-nvars:])
                i += lines_per_timestep
            else: # skip line
                i += 1
        output_arr = np.array(output_lines, dtype=float)
    df = pd.DataFrame(output_arr, columns=variables)

    name, ext = filename.split('.')
    with open(path + '/' + name + '_fixed.' + ext, 'w') as fout:
        fout.write(' = '.join(header))
        df.to_csv(fout, header=False, index=False)
