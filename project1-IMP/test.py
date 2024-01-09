import random
import multiprocessing as mp

nodes = {1, 2, 3, 4, 5, 6}
test = {}

test = nodes
test = test & {1,2,3}
print(test)
print(nodes)
