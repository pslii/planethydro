#!/home/marina/envs/planet/bin/python -u
import sys
import getopt

from planetHydro.reduceData.processCylData import processCylData
from planetHydro.reduceData.processData import processData

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'h', longopts=['x', 'xz', 'xy', 'rphi', 'xyz', 'time', 'png', 'all'])
    except getopt.GetoptError:
        print "reduceCylData --all --x --xy --rphi --xz --xyz --png --time [start] [end] [skip]"
        sys.exit(2)

    start, end, skip = None, None, None
    if len(args) == 1:
        start = int(args[0])
    elif len(args) == 2:
        start = int(args[0])
        end = int(args[1])
    elif len(args) == 3:
        start = int(args[0])
        end = int(args[1])
        skip = int(args[2])
    elif len(args) >= 4:
        print "Usage: reduceCylData --all --x --xy --rphi --xz --xyz --png --time [start] [end] [skip]"
        sys.exit(2)

    output = []
    for opt, arg in opts:
        if opt == '-h':
            print "reduceCylData -h --all --x --xy --rphi --xz --xyz --png --time"
            print "default: x, xy, time, png"
            sys.exit()
        elif opt == "--x":
            output.append('x')
        elif opt == "--xy":
            output.append('xy')
        elif opt == "--rphi":
            output.append('rphi')
        elif opt == "--xz":
            output.append('xz')
        elif opt == "--xyz":
            output.append('xyz')
        elif opt == "--time":
            output.append('time')
        elif opt == "--png":
            output.append('png')
        elif opt == "--all":
            output = ['x', 'xz', 'xy', 'rphi', 'xyz', 'time', 'png']
    if len(output) == 0:
        output = ['x', 'xy', 'time', 'png']
    processCylData(output, verbose=False, datarange=(start, end), skip=skip)


if __name__ == "__main__":
    main(sys.argv[1:])
