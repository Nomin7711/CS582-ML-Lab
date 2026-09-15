# Lab 8 - Problem 6: Solving TSP with a Genetic Algorithm
import numpy as np
import pandas as pd
import random
import operator


# A city is a point on the map; distance between two cities is Euclidean
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, other):
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self):
        return f"({self.x},{self.y})"


# Fitness = 1 / total route distance (shorter route -> higher fitness)
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0

    def route_distance(self):
        if self.distance == 0:
            path_distance = 0
            for i in range(len(self.route)):
                from_city = self.route[i]
                to_city = self.route[(i + 1) % len(self.route)]  # wraps back to start
                path_distance += from_city.distance(to_city)
            self.distance = path_distance
        return self.distance

    def route_fitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.route_distance())
        return self.fitness


# --- Encoding: a chromosome is a permutation (ordered list) of all cities ---
def create_route(city_list):
    return random.sample(city_list, len(city_list))


def initial_population(pop_size, city_list):
    return [create_route(city_list) for _ in range(pop_size)]


# Rank every route in the population by fitness (best first)
def rank_routes(population):
    fitness_results = {i: Fitness(population[i]).route_fitness() for i in range(len(population))}
    return sorted(fitness_results.items(), key=operator.itemgetter(1), reverse=True)


# --- Selection: elitism + fitness-proportionate (roulette wheel) ---
def selection(pop_ranked, elite_size):
    df = pd.DataFrame(np.array(pop_ranked), columns=["Index", "Fitness"])
    df["cum_sum"] = df.Fitness.cumsum()
    df["cum_perc"] = 100 * df.cum_sum / df.Fitness.sum()

    selection_results = [pop_ranked[i][0] for i in range(elite_size)]
    for _ in range(len(pop_ranked) - elite_size):
        pick = 100 * random.random()
        for i in range(len(pop_ranked)):
            if pick <= df.iat[i, 3]:
                selection_results.append(pop_ranked[i][0])
                break
    return selection_results


def mating_pool(population, selection_results):
    return [population[i] for i in selection_results]


# --- Crossover: ordered crossover (keeps each city exactly once) ---
def breed(parent1, parent2):
    gene_a = int(random.random() * len(parent1))
    gene_b = int(random.random() * len(parent1))
    start, end = min(gene_a, gene_b), max(gene_a, gene_b)

    child_p1 = parent1[start:end]
    child_p2 = [city for city in parent2 if city not in child_p1]
    return child_p1 + child_p2


def breed_population(matingpool, elite_size):
    pool = random.sample(matingpool, len(matingpool))
    children = matingpool[:elite_size]  # elites pass through unchanged
    for i in range(len(matingpool) - elite_size):
        children.append(breed(pool[i], pool[len(matingpool) - i - 1]))
    return children


# --- Mutation: swap two cities (keeps the permutation valid) ---
def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))
            individual[swapped], individual[swap_with] = individual[swap_with], individual[swapped]
    return individual


def mutate_population(population, mutation_rate):
    return [mutate(ind, mutation_rate) for ind in population]


def next_generation(current_gen, elite_size, mutation_rate):
    pop_ranked = rank_routes(current_gen)
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size)
    return mutate_population(children, mutation_rate)


def genetic_algorithm(city_list, pop_size, elite_size, mutation_rate, generations):
    pop = initial_population(pop_size, city_list)
    print(f"Initial distance: {1 / rank_routes(pop)[0][1]:.2f}")

    for _ in range(generations):
        pop = next_generation(pop, elite_size, mutation_rate)

    print(f"Final distance:   {1 / rank_routes(pop)[0][1]:.2f}")
    best_route_index = rank_routes(pop)[0][0]
    return pop[best_route_index]


if __name__ == "__main__":
    random.seed(42)
    city_list = [City(x=int(random.random() * 200), y=int(random.random() * 200)) for _ in range(25)]
    best_route = genetic_algorithm(city_list, pop_size=100, elite_size=20, mutation_rate=0.01, generations=500)
    print("Best route found:")
    print(best_route)
