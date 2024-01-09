import argparse
import copy
import random
import time
from queue import Queue

loop = 1  # loop time for evaluation
graph = []
n = 0
k = 0
i1 = set()
i2 = set()
s1 = set()
s2 = set()
init1_prop_set = []
init2_prop_set = []


def get_evl_arguments():
    parser = argparse.ArgumentParser(description="Heuristic for IEMP")
    # metavar: an optional parameter that specifies the name to be used for the argument's value in the help message.
    parser.add_argument("-n", metavar="<social network>", required=True, type=str, help="social network file path",
                        dest="network")
    parser.add_argument("-i", metavar="<initial seed set>", required=True, type=str, help="initial seed set file path",
                        dest="initial")
    parser.add_argument("-b", metavar="<balanced seed set>", required=True, type=str,
                        help="balanced seed set file path",
                        dest="balanced")
    parser.add_argument("-k", metavar="<budget>", type=int, required=True, help="positive integer budget", dest='k')
    return parser.parse_args()


def read_seeds(filename):
    with open(filename, 'r') as f:
        s1 = set()
        s2 = set()
        first_line = f.readline().strip("\n").split(" ")
        k1 = int(first_line[0])
        k2 = int(first_line[1])
        for i in range(k1):
            line = f.readline()
            s1.add(int(line))
        for i in range(k2):
            line = f.readline()
            s2.add(int(line))
        return s1, s2


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
            if p1 < 0.05 and p2 < 0.05:  # remove edges with low probability
                continue
            else:
                graph[u].append((v, p1, p2))
    return graph, n


def write_balance(path):
    with open(path, 'w') as f:
        f.write(str(len(s1)) + " " + str(len(s2)) + "\n")
        for v in s1:
            f.write(str(v) + "\n")
        for v in s2:
            f.write(str(v) + "\n")
    # remove the last "\n"
    with open(path, 'rb+') as f:
        f.seek(-1, 2)
        f.truncate()


def heuristic():
    global s1, s2
    s1 = set()
    s2 = set()
    while len(s1) + len(s2) <= k:
        v1, val1 = find1_max(s1, s2)
        v2, val2 = find2_max(s1, s2)
        if val1 > val2:
            s1.add(v1)
        else:
            s2.add(v2)

    return s1, s2


def find1_max(s1, s2):
    candidate = 0
    max_val = 0
    for v in range(n):
        if v not in s1:
            u1 = i1.union(s1)
            u2 = i2.union(s2)
            u1.add(v)
            val, cost = run(u1, u2)
            if val > max_val:
                max_val = val
                candidate = v
    return candidate, max_val


def find2_max(s1, s2):
    candidate = 0
    max_val = 0
    for v in range(n):
        if v not in s2:
            u1 = i1.union(s1)
            u2 = i2.union(s2)
            u2.add(v)
            val, cost = run(u1, u2)
            if val > max_val:
                max_val = val
                candidate = v
    return candidate, max_val


def run_initial():
    for i in range(loop):
        init1_prop_set.append(ic_process(i1, 1))
        init2_prop_set.append(ic_process(i2, 2))


def ic_process(seeds, cam):  # independent cascade
    q = Queue()
    active = copy.deepcopy(seeds)
    exposed = copy.deepcopy(seeds)
    for seed in seeds:
        q.put(seed)
    while not q.empty():  # graph[u] = [(v, p1, p2) ...]
        u = q.get()
        for nei in graph[u]:  # u's adjacent nodes
            v = nei[0]
            p = nei[cam]  # p1 or p2
            exposed.add(v)
            if v not in active:
                prob = random.random()
                if p >= prob:  # activate v
                    active.add(v)
                    q.put(v)
    return exposed, active


def ic_both(u1, u2):  # run ic for 2 campaigns
    exp1, act1 = ic_process(u1, 1)
    exp2, act2 = ic_process(u2, 2)
    num = len(exp1.union(exp2)) - len(exp1.intersection(exp2))
    return n - num


def run(u1, u2):  # run ic for 2 campaigns
    start = time.perf_counter()
    total = 0
    for i in range(loop):
        total += ic_both(u1, u2)

    end = time.perf_counter()
    return total / loop, end - start


if __name__ == "__main__":
    args = get_evl_arguments()
    network = args.network  # network file path
    initial = args.initial
    balanced = args.balanced  # output file path
    k = args.k  # budget
    graph, n = read_network(network)
    i1, i2 = read_seeds(initial)
    start = time.perf_counter()
    s1, s2 = heuristic()
    end = time.perf_counter()
    write_balance(balanced)
    # print("Time: ", end - start)
