from enhanced_aco import EnhancedACO
import csv
from datetime import datetime
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

if __name__ == '__main__':
    nodes = load_location_data('tsp dataset/location_ll.txt')

    aco = EnhancedACO(
        nodes=nodes,
        mode='ACS',
        colony_size=15,
        steps=100,
        alpha=1.0,
        beta=3.0,
        rho=0.1,
        distance_metric='haversine',
        noise_level=0.05,
        max_time=10
    )

    runtime, distance = aco.run()
    print(f'Runtime: {runtime:.2f}s')
    print(f'Distance: {distance:.2f}')

    aco.plot_best_tour()
    aco.plot_convergence()

    # Save results to CSV
    csv_file = 'results/enhanced_run_log.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(
                ['Timestamp', 'Mode', 'Colony Size', 'Steps', 'Alpha', 'Beta', 'Rho', 'Noise Level', 'Distance Metric',
                 'Runtime (s)', 'Distance'])

        writer.writerow([
            datetime.now(),
            aco.mode,
            aco.colony_size,
            aco.steps,
            aco.alpha,
            aco.beta,
            aco.rho,
            aco.noise_level,
            aco.distance_metric,
            round(runtime, 4),
            round(distance, 4)
        ])