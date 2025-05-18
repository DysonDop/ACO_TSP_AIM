import csv
import time
from datetime import datetime
from enhanced_aco import EnhancedACO
import os

def load_location_data(file_path):
    location = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                lat, lon = map(float, parts)
                location.append((lat, lon))
    return location

# Setup
nodes = load_location_data('tsp dataset/location_ll.txt')
modes = ['ACS', 'Elitist', 'MaxMin']
colony_size = 15
steps = 100
alpha = 1.0
beta = 3.0
rho = 0.1
noise_level = 0.05
distance_metric = 'haversine'

# Prepare CSV logging
file_path = 'results/enhanced_comparison_log.csv'
file_exists = os.path.isfile(file_path)

with open(file_path, 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write header only once
    if not file_exists:
        writer.writerow([
            'Timestamp', 'Mode', 'Trial', 'Colony Size', 'Steps', 'Alpha', 'Beta', 'Rho',
            'Noise Level', 'Distance Metric', 'Runtime (s)', 'Distance'
        ])

    # Repeat each mode 10 times
    for mode in modes:
        for trial in range(10):
            print(f"[{mode}] Trial {trial+1}/10")

            aco = EnhancedACO(
                nodes=nodes,
                mode=mode,
                colony_size=colony_size,
                steps=steps,
                alpha=alpha,
                beta=beta,
                rho=rho,
                distance_metric=distance_metric,
                noise_level=noise_level,
                max_time=10
            )

            runtime, distance = aco.run()

            writer.writerow([
                datetime.now(), mode, trial + 1, colony_size, steps, alpha, beta, rho,
                noise_level, distance_metric, round(runtime, 4), round(distance, 4)
            ])