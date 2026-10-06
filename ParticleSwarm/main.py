import numpy as np
import matplotlib.pyplot as plt

LAMBDA = np.array([0.25, 0.35, 0.20, 0.30])  
SATURATION_FLOW = 0.8  
LOST_TIME = 4.0       
CYCLE_MIN = 40.0       
CYCLE_MAX = 180.0      
GREEN_MIN = 10.0       

def objective_function(green_times):
    """
    Evaluates traffic network penalty based on M/M/1 queuing theory model.
    Goal: Minimize average queue length and delay per cycle.
    """
    C = np.sum(green_times) + len(green_times) * LOST_TIME  
    
    if C < CYCLE_MIN or C > CYCLE_MAX:
        return 1e6 + abs(C - CYCLE_MIN) * 1000
    
    total_delay = 0.0
    for i, g in enumerate(green_times):
        effective_capacity = SATURATION_FLOW * (g / C) 
        arrival_rate = LAMBDA[i]
        
        if arrival_rate >= effective_capacity:
            return 1e5 + (arrival_rate - effective_capacity) * 50000
        
        rho = arrival_rate / effective_capacity
        avg_queue_length = rho / (1.0 - rho)
        
        d1 = (C * (1 - g / C)**2) / (2 * (1 - min(rho, 0.95) * (g / C)))
        total_delay += avg_queue_length + d1

    return total_delay


def run_pso(num_particles=30, max_iter=100, w=0.7, c1=1.5, c2=1.5):
    num_dimensions = len(LAMBDA)
    
    bounds_min = np.full(num_dimensions, GREEN_MIN)
    bounds_max = np.full(num_dimensions, (CYCLE_MAX - len(LAMBDA) * LOST_TIME) / num_dimensions)
    
    positions = np.random.uniform(bounds_min, bounds_max, (num_particles, num_dimensions))
    velocities = np.random.uniform(-1, 1, (num_particles, num_dimensions))
    
    pbest_positions = np.copy(positions)
    pbest_scores = np.array([objective_function(p) for p in positions])
    
    gbest_index = np.argmin(pbest_scores)
    gbest_position = np.copy(pbest_positions[gbest_index])
    gbest_score = pbest_scores[gbest_index]
    
    history = [gbest_score]

    for iteration in range(max_iter):
        r1 = np.random.rand(num_particles, num_dimensions)
        r2 = np.random.rand(num_particles, num_dimensions)
        
        velocities = (w * velocities 
                      + c1 * r1 * (pbest_positions - positions) 
                      + c2 * r2 * (gbest_position - positions))
        
        positions += velocities
        
        positions = np.clip(positions, bounds_min, bounds_max)
        
        for i in range(num_particles):
            score = objective_function(positions[i])
            
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = np.copy(positions[i])
                
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = np.copy(positions[i])
                    
        history.append(gbest_score)
        
        if (iteration + 1) % 20 == 0 or iteration == 0:
            print(f"Iteration {iteration + 1:3d}/{max_iter} | Best Cost (Delay): {gbest_score:.4f}")

    return gbest_position, gbest_score, history



if __name__ == "__main__":
    print("--- Running Traffic Signal PSO Optimizer ---")
    best_green_times, best_cost, convergence_history = run_pso()
    
    total_cycle = np.sum(best_green_times) + len(best_green_times) * LOST_TIME
    
    print("\n================ OPTIMIZATION RESULTS ================")
    print(f"Optimal Total Cycle Time: {total_cycle:.2f} seconds")
    for i, g in enumerate(best_green_times):
        print(f"  Phase {i+1} Green Time: {g:.2f} sec (Arrival rate: {LAMBDA[i]} veh/s)")
    print(f"Minimized Traffic Delay Score: {best_cost:.4f}")
    print("======================================================")

    plt.figure(figsize=(8, 4))
    plt.plot(convergence_history, color='blue', linewidth=2, label="Best Fitness")
    plt.title("PSO Convergence Curve for Traffic Signal Optimization")
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function Value (Delay / Queue Penalty)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
