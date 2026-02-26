import os 
import sys 
import numpy as np 
from simulation_utils import create_env 

def sample_uniform(bounds, rng): 
    """ bounds: list of (low, high) pairs of length feed_size. """ 
    lows = np.array([b[0] for b in bounds], dtype=float) 
    highs = np.array([b[1] for b in bounds], dtype=float) 
    return rng.uniform(lows, highs) 

def compute_features(sim, u): 
    sim.feed(u) 
    # Many models compute features from the most recent rollout 
    phi = sim.get_features() 
    return np.asarray(phi, dtype=float) 

def main(): 
    if len(sys.argv) < 3: 
        print("Usage: python item_sampler.py <TaskName> <N> [seed]") 
        print("Example: python item_sampler.py Driver 500 0") 
        sys.exit(1) 
    
    task = sys.argv[1] 
    N = int(sys.argv[2]) 
    seed = int(sys.argv[3]) if len(sys.argv) >= 4 else 0 
    rng = np.random.default_rng(seed) 
    sim = create_env(task) 
    feed_size = sim.feed_size 
    d = sim.num_of_features 
    
    # Ensure output directory exists 
    out_dir = "ctrl_samples" 
    os.makedirs(out_dir, exist_ok=True) 
    out_path = os.path.join(out_dir, f"{task.lower()}_items.npz") 

    u_set = np.zeros((N, feed_size), dtype=float) 
    phi_raw = np.zeros((N, d), dtype=float) 

    for i in range(N): 
        u = sample_uniform(sim.feed_bounds, rng) 
        u_set[i, :] = u 
        phi_raw[i, :] = compute_features(sim, u) 
        if (i + 1) % max(1, N // 10) == 0: 
            print(f"[item_sampler] {i+1}/{N} items generated") 

    # Normalize features for MaxInP computations (recommended) 
    phi_mean = phi_raw.mean(axis=0) 
    phi_std = phi_raw.std(axis=0) 
    # avoid divide-by-zero 
    phi_std = np.where(phi_std < 1e-8, 1.0, phi_std) 
    phi_norm = (phi_raw - phi_mean) / phi_std 

    np.savez( 
        out_path, 
        task=task, 
        feed_size=feed_size, 
        num_features=d, 
        u_set=u_set, 
        phi_raw=phi_raw, 
        phi_norm=phi_norm, 
        phi_mean=phi_mean, 
        phi_std=phi_std, 
        seed=seed, 
    ) 
if __name__ == "__main__": 

    main() 