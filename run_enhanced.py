from enhanced_aco import EnhancedACO

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
        max_time=10  # Optional, can be None
    )

    runtime, distance = aco.run()
    print(f'Runtime: {runtime:.2f}s')
    print(f'Distance: {distance:.2f}')

    aco.plot_best_tour()
    aco.plot_convergence()
