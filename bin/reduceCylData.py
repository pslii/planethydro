#!/home/marina/envs/planet/bin/python -u
import sys
import getopt

from planetHydro.reduceData.processCylData import processCylData

def main(argv):
    try:
        opts, args = getopt.getopt(argv, 'h', longopts=['x', 'xz', 'xy', 'xyz', 'time', 'png', 'all'])
    except getopt.GetoptError:
        print "reduceCylData --all --x --xy --xz --xyz --png --time"
        sys.exit(2)
    output = []
    for opt, arg in opts:
        if opt == '-h':
            print "reduceCylData -h --all --x --xy --xz --xyz --png --time"
            sys.exit()
        elif opt == "--x":
            output.append('x')
        elif opt == "--xy":
            output.append('xy')
        elif opt == "--xz":
            output.append('xz')
        elif opt == "--xyz":
            output.append('xyz')
        elif opt == "--time":
            output.append('time')
        elif opt == "--png":
            output.append('png')
        elif opt == "--all":
            output = ['x', 'xz', 'xy', 'xyz', 'time', 'png']
    if len(output) == 0:
        output = ['x', 'xy', 'xz', 'xyz', 'time']
    processCylData(output, verbose=False)


if __name__ == "__main__":
    main(sys.argv[1:])