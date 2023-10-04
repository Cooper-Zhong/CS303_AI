import argparse
import random
import time
import multiprocessing as mp

import numpy as np
from queue import Queue


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
    return graph, n


# read seed set file
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


# cam: campaign, 1 or 2, to get p1 or p2.
def ic_process(graph, seeds, cam):  # independent cascade
    q = Queue()
    active = set()
    for seed in seeds:
        q.put(seed)
        active.add(seed)
    while not q.empty():  # graph[u] = [(v, p1, p2) ...]
        u = q.get()
        for nei in graph[u]:  # u's adjacent nodes
            v = nei[0]
            p = nei[cam]  # p1 or p2
            if v not in active:
                prob = random.random()
                if p >= prob:  # activate v
                    active.add(v)
                    q.put(v)
    return active


def ic_2cam(graph, u1, u2, n):  # run ic for 2 campaigns
    act1 = ic_process(graph, u1, 1)
    act2 = ic_process(graph, u2, 2)
    num = len(act1.union(act2)) - len(act1.intersection(act2))
    return n - num


def run(graph, u1, u2, n):
    loop = 5000
    cores = 4
    pool = mp.Pool(processes=cores)
    results = []
    start = time.time()
    for _ in range(loop):
        res = pool.apply_async(ic_2cam, args=(graph, u1, u2, n))
        results.append(res)

    pool.close()
    pool.join()
    total_return_val = sum(res.get() for res in results)
    ans = total_return_val / loop
    end = time.time()
    print("result: ", ans)
    print("time: ", end - start)


if __name__ == "__main__":
    args = get_evl_arguments()
    network = args.network
    initial = args.initial
    balanced = args.balanced
    k = args.k
    out = args.out
    graph, n = read_network(network)
    i1, i2 = read_seeds(initial)
    s1, s2 = read_seeds(balanced)
    u1 = i1.union(s1)
    u2 = i2.union(s2)
    # for i in range(3):
    run(graph, u1, u2, n)

    # case1: 322
    # case2: 35985