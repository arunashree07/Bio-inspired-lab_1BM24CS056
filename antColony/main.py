import numpy as np


class RobotACORouter:

    def __init__(
        self,
        waypoints,
        n_ants=20,
        iterations=100,
        alpha=1.0,
        beta=2.0,
        evaporation=0.5,
        Q=100,
    ):
        """Initializes the ACO router for robotic waypoint navigation.

        Args:
            waypoints (list of tuples): Coordinate list [(x1, y1), (x2, y2),
            ...]
            n_ants (int): Number of artificial ants exploring per iteration.
            iterations (int): Total training loops.
            alpha (float): Relative importance of pheromone trail.
            beta (float): Relative importance of distance proximity (heuristic).
            evaporation (float): Rate at which pheromones decay (0 to 1).
            Q (float): Pheromone deposition constant.
        """
        self.waypoints = np.array(waypoints)
        self.n_nodes = len(waypoints)
        self.n_ants = n_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = Q

        # Generate the distance matrix between all target nodes
        self.dist_matrix = self._calculate_distance_matrix()

        # Initialize pheromone matrix with small positive values
        self.pheromone = np.ones((self.n_nodes, self.n_nodes)) / self.n_nodes

    def _calculate_distance_matrix(self):
        # Computes the Euclidean distance between every single waypoint pair
        X = self.waypoints[:, 0][:, np.newaxis]
        Y = self.waypoints[:, 1][:, np.newaxis]
        dist = np.sqrt((X - X.T) ** 2 + (Y - Y.T) ** 2)
        # Avoid division-by-zero on self-loops by replacing 0 with infinity
        np.fill_diagonal(dist, np.inf)
        return dist

    def _select_next_node(self, current_node, visited):
        # Compute heuristic desirability (inverse of distance)
        eta = 1.0 / self.dist_matrix[current_node]
        tau = self.pheromone[current_node]

        # Standard state transition probability calculation
        probabilities = (tau**self.alpha) * (eta**self.beta)

        # Force probabilities of already visited waypoints to zero
        probabilities[list(visited)] = 0

        # Avoid NaN values if all remaining paths evaluate to 0
        if probabilities.sum() == 0:
            unvisited = [i for i in range(self.n_nodes) if i not in visited]
            return np.random.choice(unvisited)

        probabilities /= probabilities.sum()
        return np.random.choice(range(self.n_nodes), p=probabilities)

    def optimize(self):
        best_path = None
        best_distance = float("inf")

        for iteration in range(self.iterations):
            all_paths = []
            all_distances = []

            for ant in range(self.n_ants):
                # Every robot/ant starts at the initial waypoint (Base node 0)
                current_node = 0
                visited = {current_node}
                path = [current_node]

                while len(visited) < self.n_nodes:
                    next_node = self._select_next_node(current_node, visited)
                    path.append(next_node)
                    visited.add(next_node)
                    current_node = next_node

                # Loop the robot back to the starting position
                path.append(path[0])

                # Calculate total journey distance
                distance = sum(
                    self.dist_matrix[path[i], path[i + 1]]
                    for i in range(self.n_nodes)
                )

                all_paths.append(path)
                all_distances.append(distance)

                # Capture global best solution
                if distance < best_distance:
                    best_distance = distance
                    best_path = path

            # Evaporate current pheromones across the entire grid
            self.pheromone *= 1.0 - self.evaporation

            # Update pheromones based on the paths traveled in this iteration
            for path, distance in zip(all_paths, all_distances):
                for i in range(self.n_nodes):
                    node_a = path[i]
                    node_b = path[i + 1]
                    # Shorter paths yield a higher concentration contribution
                    self.pheromone[node_a, node_b] += self.Q / distance
                    self.pheromone[node_b, node_a] += self.Q / distance

        return best_path, best_distance


if __name__ == "__main__":
    robot_targets = [(0, 0), (2, 5), (3, 1), (6, 4), (1, 8), (7, 2)]

    router = RobotACORouter(
        waypoints=robot_targets, n_ants=15, iterations=50, alpha=1.0, beta=2.0
    )
    optimal_path, optimal_dist = router.optimize()

    print(f"Optimal Waypoint Routing Sequence: {optimal_path}")
    print(f"Total Traversed Path Distance: {optimal_dist:.2f} units")
