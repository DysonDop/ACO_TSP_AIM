import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class EnhancedACO:
    def __init__(self, nodes, mode='ACS', colony_size=10, steps=100, alpha=1.0, beta=3.0, rho=0.1,
                 distance_metric='euclidean', noise_level=0.0, max_time=None):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        self.mode = mode
        self.colony_size = colony_size
        self.steps = steps
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.initial_alpha = alpha
        self.initial_beta = beta
        self.initial_rho = rho
        self.distance_metric = distance_metric
        self.noise_level = noise_level
        self.max_time = max_time

        self.pheromone = np.ones((self.num_nodes, self.num_nodes))
        self.distances = self.compute_distances()
        self.global_best_tour = None
        self.global_best_distance = float('inf')
        self.distance_history = []

    def compute_distances(self):
        distances = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    dist = self.calculate_distance(self.nodes[i], self.nodes[j])
                    if self.noise_level > 0:
                        noise = random.uniform(-self.noise_level, self.noise_level)
                        dist *= (1 + noise)
                    # Fix zero-distance bug
                    if dist == 0:
                        dist = 1e-10
                    distances[i][j] = dist
        return distances

    def calculate_distance(self, a, b):
        if self.distance_metric == 'manhattan':
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        elif self.distance_metric == 'haversine':
            R = 6371
            lat1, lon1 = math.radians(a[0]), math.radians(a[1])
            lat2, lon2 = math.radians(b[0]), math.radians(b[1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            return R * 2 * math.asin(math.sqrt(hav))
        else:
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def run(self):
        start_time = time.time()
        for step in range(self.steps):
            if self.max_time and (time.time() - start_time > self.max_time):
                break

            alpha = self.initial_alpha + (step / self.steps) * (2.0 - self.initial_alpha)
            beta = self.initial_beta + (step / self.steps) * (5.0 - self.initial_beta)
            rho = self.initial_rho + (step / self.steps) * (0.5 - self.initial_rho)

            all_tours = []
            all_distances = []

            for _ in range(self.colony_size):
                tour = self.construct_tour(alpha, beta)
                distance = self.calculate_tour_distance(tour)
                all_tours.append(tour)
                all_distances.append(distance)

                if distance < self.global_best_distance:
                    self.global_best_distance = distance
                    self.global_best_tour = tour

            self.update_pheromones(all_tours, all_distances, rho)
            self.distance_history.append(self.global_best_distance)

        return time.time() - start_time, self.global_best_distance

    def construct_tour(self, alpha, beta):
        tour = [random.randint(0, self.num_nodes - 1)]
        unvisited = set(range(self.num_nodes)) - set(tour)

        while unvisited:
            current = tour[-1]
            probabilities = []
            for node in unvisited:
                pheromone = self.pheromone[current][node] ** alpha
                heuristic = (1 / self.distances[current][node]) ** beta
                probabilities.append(pheromone * heuristic)

            total = sum(probabilities)
            probabilities = [p / total for p in probabilities]
            chosen = random.choices(list(unvisited), weights=probabilities, k=1)[0]
            tour.append(chosen)
            unvisited.remove(chosen)

        return tour

    def calculate_tour_distance(self, tour):
        return sum(self.distances[tour[i]][tour[(i + 1) % self.num_nodes]] for i in range(self.num_nodes))

    def update_pheromones(self, tours, distances, rho):
        self.pheromone *= (1 - rho)
        for tour, dist in zip(tours, distances):
            for i in range(self.num_nodes):
                a, b = tour[i], tour[(i + 1) % self.num_nodes]
                self.pheromone[a][b] += 1 / dist
                self.pheromone[b][a] += 1 / dist

    def plot_best_tour(self):
        tour = self.global_best_tour + [self.global_best_tour[0]]
        x = [self.nodes[i][1] for i in tour]  # Longitude
        y = [self.nodes[i][0] for i in tour]  # Latitude

        plt.figure(figsize=(10, 6))

        # Load and display background map
        try:
            img = mpimg.imread('assets/map.png')
            min_lon = min(x)
            max_lon = max(x)
            min_lat = min(y)
            max_lat = max(y)
            padding = 2  # degrees of padding
            plt.imshow(
                img,
                extent = [min_lon - padding, max_lon + padding, min_lat - padding, max_lat + padding],
                alpha=0.6
            )
        except FileNotFoundError:
            print("⚠️ Background map not found. Showing tour without map.")

        # Plot the path
        plt.plot(x, y, marker='o', linestyle='-', color='blue', linewidth=2)
        plt.scatter(x, y, color='red', s=40)
        plt.title(f'ACO Best Tour ({self.mode}) - Distance: {round(self.global_best_distance, 2)}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.savefig('results/best_tour.png', dpi=300)
        plt.show()

    def plot_convergence(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.distance_history)
        plt.title('Convergence Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.grid(True)
        plt.savefig('results/convergence_plot.png', dpi=300)
        plt.show()

        def plot_best_tour(self):
            tour = self.global_best_tour + [self.global_best_tour[0]]
            x = [self.nodes[i][1] for i in tour]  # Longitude
            y = [self.nodes[i][0] for i in tour]  # Latitude

            plt.figure(figsize=(10, 6))

            # Load and display background map
            try:
                img = mpimg.imread('assets/map.png')
                plt.imshow(img, extent=[-180, 180, -90, 90],
                           alpha=0.6)  # Adjust extent to match your scaled coordinates
            except FileNotFoundError:
                print("⚠️ Background map not found. Showing tour without map.")

            # Plot the path
            plt.plot(x, y, marker='o', linestyle='-', color='blue', linewidth=2)
            plt.scatter(x, y, color='red', s=40)
            plt.title(f'ACO Best Tour ({self.mode}) - Distance: {round(self.global_best_distance, 2)}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.grid(True)
            plt.show()