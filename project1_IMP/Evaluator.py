import argparse
import numpy as np


# read network file
def read_network(filename):
    with open(filename, 'r') as f:
        first_line = f.readline().strip("\n").split(" ")
        n = int(first_line[0])
        m = int(first_line[1])
        graph = [[] for _ in range(n + 1)]  # adj list

        lines = f.readlines()
        for line in lines:
            line = line.strip(" ")
            # u->v, p1, p2
            u = int(line.split(" ")[0])
            v = int(line.split(" ")[1])
            p1 = float(line.split(" ")[2])
            p2 = float(line.split(" ")[3])
            graph[u].append((v, p1, p2))
        return graph


# read seed initial set file
def read_initial(filename):
    with open(filename, 'r') as f:
        I1 = []
        I2 = []
        first_line = f.readline().strip("\n").split(" ")
        k1 = int(first_line[0])
        k2 = int(first_line[1])
        for i in range(k1):
            line = f.readline()
            I1.append(int(line))
        for i in range(k2):
            line = f.readline()
            I2.append(int(line))
        return I1, I2


# read balanced seed set file
def read_balanced(filename):
    with open(filename, 'r') as f:
        s1 = []
        s2 = []
        first_line = f.readline().strip("\n").split(" ")
        k1 = int(first_line[0])
        k2 = int(first_line[1])
        for i in range(k1):
            line = f.readline()
            s1.append(int(line))
        for i in range(k2):
            line = f.readline()
            s2.append(int(line))
        return s1, s2


# read arguments
def get_evl_arguments():
    parser = argparse.ArgumentParser(description="Evaluator for IMP")
    # metavar: an optional parameter that specifies the name to be used for the argument's value in the help message.
    parser.add_argument("-n", metavar="<social network>", required=True, type=str, help="social network file path",
                        dest="network")
    parser.add_argument("-i", metavar="<initial seed set>", required=True, type=str, help="initial seed set file path",
                        dest="initial")
    parser.add_argument("-b", metavar="<balanced seed set>", required=True, type=str,
                        help="balanced seed set file path",
                        dest="balanced")
    parser.add_argument("-k", metavar="<budget>", type=int, required=True, help="positive integer budget", dest='k')
    parser.add_argument("-o", metavar="<object value output path>", type=str, required=True,
                        help="objective value output path", dest="out")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_evl_arguments()
    network = args.network
    initial = args.initial
    balanced = args.balanced
    k = args.k
    out = args.out
    graph = read_network(network)
    i1, i2 = read_initial(initial)
    s1, s2 = read_balanced(balanced)
    print("initial 1: ", i1)
    print("initial 2: ", i2)
    print("balanced 1: ", s1)
    print("balanced 2: ", s2)
    for i in range(len(graph[0])):
        print(graph[0][i])

