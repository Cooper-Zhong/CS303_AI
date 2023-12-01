import argparse
import copy
import random
import time

sample_num = 3  # loop time for evaluation
n = 0
k = 0
evol_num = 40
pop_size = 30
cross_rate = 0.5
mutation_rate = 0.3
graph = []
nei = []
nodes = set()
pre_exposed1 = []
pre_exposed2 = []
i1 = set()
i2 = set()
s1 = set()
s2 = set()

exposed_init1 = set()
exposed_init2 = set()


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


def uniform_crossover(parent1, parent2):
    child1 = [parent1[i] if random.random() < cross_rate else parent2[i] for i in range(len(parent1))]
    child2 = [parent2[i] if random.random() < cross_rate else parent1[i] for i in range(len(parent1))]
    return child1, child2


def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def flip_bit(solution):
    mutated_solution = solution.copy()
    for i in range(len(mutated_solution)):
        if random.random() < mutation_rate:
            mutated_solution[i] = mutated_solution[i] ^ 1
    return mutated_solution


def generate_offspring(population):
    offspring = []
    num = len(population) * 2
    while len(offspring) < num:
        # Select two parents from the population
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        # parent1 = roulette_wheel_selection(population)
        # parent2 = roulette_wheel_selection(population)

        child1, child2 = uniform_crossover(parent1, parent2)
        # child1, child2 = single_point_crossover(parent1, parent2)

        # Perform flip bit mutation
        child1 = flip_bit(child1)
        child2 = flip_bit(child2)

        # Add the child to the offspring
        offspring.append(child1)
        offspring.append(child2)

    return offspring


def seq_to_set(solution):
    set1 = set()
    set2 = set()
    for i in range(n):
        if solution[i] == 1:
            set1.add(i)
        if solution[n + i] == 1:
            set2.add(i)
    return set1, set2


def get_fitness(solution):
    # Calculate the fitness of a solution by counting the number of ones
    cnt = sum(solution)
    if cnt > k:  # invalid solution
        return -1 * cnt
    set1, set2 = seq_to_set(solution)
    exposed1 = set()
    exposed2 = set()
    for i in set1:  # s1 的影响集
        exposed1 = exposed1 | pre_exposed1[i]
    for i in set2:
        exposed2 = exposed2 | pre_exposed2[i]
    return get_phi(exposed1 | exposed_init1, exposed2 | exposed_init2)


def roulette_wheel_selection(population):
    fitness_values = [get_fitness(individual) for individual in population]
    total_fitness = sum(fitness_values)

    selection_probabilities = [fitness / total_fitness for fitness in fitness_values]
    selected_population = []

    for _ in range(pop_size):
        # Randomly choose a value on the wheel
        roulette_value = random.uniform(0, 1)

        # Select an individual based on the roulette value
        cumulative_probability = 0
        for i, probability in enumerate(selection_probabilities):
            cumulative_probability += probability
            if cumulative_probability >= roulette_value:
                selected_population.append(population[i])
                break

    return selected_population


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

    exposed_init1 = get_exposed(i1, 1, sample_num)  # exposed set of i1
    exposed_init2 = get_exposed(i2, 2, sample_num)  # exposed set of i2

    start = time.perf_counter()

    # initial population
    nodes_not_in1 = nodes - i1
    nodes_not_in2 = nodes - i2

    income1 = [(0, i, 0) for i in range(n)]
    income2 = [(1, i, 0) for i in range(n)]

    # calculate income
    for node in nodes_not_in1:
        # exposed set after adding node
        temp_exposed = exposed_init1 | pre_exposed1[node]
        val = get_phi(temp_exposed, exposed_init2)
        income1[node] = (0, node, val)

    for node in nodes_not_in2:
        temp_exposed = exposed_init2 | pre_exposed2[node]
        val = get_phi(temp_exposed, exposed_init1)
        income2[node] = (1, node, val)

    # sort income
    income = income1 + income2
    income.sort(key=lambda x: x[2], reverse=True)

    # Generate a population with n solutions randomly
    population = []
    for i in range(pop_size):
        # solution[i] = 1 means i is selected, first n nodes are cam1, last n nodes are cam2
        solution = [0 for _ in range(2 * n)]
        # randomly select 0-k nodes from top n/2 nodes

        selected_elements = random.sample(income[:n // 2], random.randint(0, k - 1))
        for i in range(len(selected_elements)):
            if selected_elements[i][0] == 0:  # cam1
                solution[selected_elements[i][1]] = 1
                population.append(solution)
            else:
                solution[n + selected_elements[i][1]] = 1  # cam2
                population.append(solution)

    # evolutionary ==========================================================

    # Generate a population with n solutions randomly
    # population = []
    # for i in range(pop_size):
    #     # solution[i] = 1 means i is selected, first n nodes are cam1, last n nodes are cam2
    #     solution = [0 for _ in range(2 * n)]
    #     # randomly select 0-k positions to set 1
    #     selected_elements = random.sample(range(2 * n), random.randint(0, k))
    #     for i in range(len(selected_elements)):
    #         solution[selected_elements[i]] = 1
    #     population.append(solution)
    #
    # population = roulette_wheel_selection(population)

    best = random.choice(population)
    best_fitness = get_fitness(best)
    for _ in range(evol_num):  # evolution loop
        # update initial estimation with a low probability
        if random.random() < 0.15:
            exposed_init1 = get_exposed(i1, 1, sample_num)  # exposed set of i1
            exposed_init2 = get_exposed(i2, 2, sample_num)  # exposed set of i2

        # Generate n new solutions by crossover and mutation
        offspring = generate_offspring(population)

        candidates = []
        # Evaluate the fitness of the new solutions
        for i in range(len(population)):
            candidates.append((population[i], get_fitness(population[i])))
        for i in range(len(offspring)):
            candidates.append((offspring[i], get_fitness(offspring[i])))

        # Select the best n solutions from the population and the offspring
        candidates.sort(key=lambda x: x[1], reverse=True)

        best = random.choices(candidates[:3], weights=[0.7, 0.2, 0.1], k=1)[0][0]
        best_fitness = get_fitness(best)

        print(f"round {_ + 1}: ", best_fitness)  # print best fitness(phi)

        population = [candidates[i][0] for i in range(pop_size)]  # select the best n solutions

    # select the best solution
    s1, s2 = seq_to_set(best)

    end = time.perf_counter()
    print("time: ", end - start)
    # write output
    with open(args.output, 'w') as f:
        f.write(str(len(s1)) + " " + str(len(s2)) + "\n")
        for v in s1:
            f.write(str(v) + "\n")
        for v in s2:
            f.write(str(v) + "\n")
