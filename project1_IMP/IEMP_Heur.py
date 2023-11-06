import argparse
import copy
import random
import time

sample_num = 3  # loop time for evaluation
n = 0
k = 0
graph = []
nei = []
nodes = set()
i1 = set()
i2 = set()
s1 = set()
s2 = set()


def ic_process(seeds, cam, nei):  # independent cascade
    q = copy.deepcopy(seeds)
    active = copy.deepcopy(seeds)
    exposed = copy.deepcopy(seeds)
    while q:  # graph[u] = [(v, p1, p2) ...]
        u = q.pop()
        exposed = exposed | nei[u]
        for next in graph[u]:  # u's adjacent nodes
            v = next[0]
            if v not in active:
                p = next[cam]  # p1 or p2
                prob = random.random()
                if p >= prob:  # activate v
                    active.add(v)
                    q.add(v)
    return exposed


def get_exposed(seeds, cam, sample_times):
    exposed = nodes
    for i in range(sample_times):
        exposed = exposed & ic_process(seeds, cam, nei)
    return exposed


def get_phi(exposed1, exposed2):
    return n - len(exposed1 ^ exposed2)  # symmetric difference


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # metavar: an optional parameter that specifies the name to be used for the argument's value in the help message.
    parser.add_argument("-n", required=True, type=str, help="social network file path",
                        dest="network")
    parser.add_argument("-i", metavar="<initial seed set>", required=True, type=str, help="initial seed set file path",
                        dest="initial")
    parser.add_argument("-b", metavar="<balanced seed set>", required=True, type=str,
                        help="balanced seed set file path",
                        dest="output")
    parser.add_argument("-k", metavar="<budget>", type=int, required=True, help="positive integer budget",
                        dest='budget')
    args = parser.parse_args()

    # read network
    with open(args.network, 'r') as f:
        first_line = f.readline().strip("\n").split(" ")
        n = int(first_line[0])
        m = int(first_line[1])
        graph = [[] for _ in range(n + 1)]  # adj list
        nei = [set() for _ in range(n + 1)]

        line = f.readline()
        while line:
            line = line.strip("\n")
            line = line.split(" ")
            # u->v, p1, p2
            u = int(line[0])
            v = int(line[1])
            p1 = float(line[2])
            p2 = float(line[3])
            nei[u].add(v)

            if p1 >= 0.05 or p2 >= 0.05:  # remove edges with low probability
                graph[u].append((v, p1, p2))

            line = f.readline()

    # read seeds
    with open(args.initial, 'r') as f:
        first_line = f.readline().strip("\n").split(" ")
        k1 = int(first_line[0])
        k2 = int(first_line[1])
        for i in range(k1):
            line = f.readline()
            i1.add(int(line))
        for i in range(k2):
            line = f.readline()
            i2.add(int(line))
    # read budget
    k = args.budget

    nodes = set(range(n))  # all nodes
    # each single node's exposed set
    pre_exposed1 = [get_exposed({i}, 1, sample_num) for i in range(n)]
    pre_exposed2 = [get_exposed({i}, 2, sample_num) for i in range(n)]
    s1 = set()
    s2 = set()
    start = time.perf_counter()

    while len(s1) + len(s2) < k:
        u1 = i1 | s1
        u2 = i2 | s2
        exposed_1 = get_exposed(u1, 1, sample_num)
        exposed_2 = get_exposed(u2, 2, sample_num)

        max1 = 0
        max2 = 0
        candidate1 = 0
        candidate2 = 0
        nodes_not_in1 = nodes - u1
        nodes_not_in2 = nodes - u2
        for node in nodes_not_in1:  # nodes not in i1 or s1
            temp_exposed = exposed_1 | pre_exposed1[node]  # exposed set after adding node
            val = get_phi(temp_exposed, exposed_2)
            if val > max1:
                max1 = val
                candidate1 = node

        for node in nodes_not_in2:
            temp_exposed = exposed_2 | pre_exposed2[node]
            val = get_phi(exposed_1, temp_exposed)
            if val > max2:
                max2 = val
                candidate2 = node

        if max1 > max2:
            s1.add(candidate1)
        else:
            s2.add(candidate2)

    end = time.perf_counter()
    print("time: ", end - start)
    # write output
    with open(args.output, 'w') as f:
        f.write(str(len(s1)) + " " + str(len(s2)) + "\n")
        for v in s1:
            f.write(str(v) + "\n")
        for v in s2:
            f.write(str(v) + "\n")
