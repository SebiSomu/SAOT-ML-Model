"""
Coordinates: (x, y) where x = depth on field (0=own goal line, 100=opponent goal)
             y = width of field (0=left, 100=right)

Offside Logic (simplified, phase 1):
  - The teammate (co-player) is in offside if their x position
    is greater than the defender's x position at the moment of the pass.
  - Only x position is considered (depth), not y.

Three Players:
  1. Passer (Forward): The player who passes the ball
  2. Teammate (Co-player): The player receiving the pass - can be offside
  3. Defender: The last opponent who could catch the teammate
"""

import numpy as np
import pandas as pd


def generate_offside_sample(n_samples: int = 2000, seed: int = 42):
    """
    Generate synthetic data for offside detection with 3 players.

    Features returned (per sample):
        - passer_x     : x position of the passer (forward)
        - passer_y     : y position of the passer
        - teammate_x   : x position of the teammate (co-player)
        - teammate_y   : y position of the teammate
        - defender_x   : x position of the last defender
        - defender_y   : y position of the defender
        - x_diff       : teammate_x - defender_x (main feature)

    Label:
        - offside: 1 if teammate is in offside, 0 otherwise
    """
    rng = np.random.default_rng(seed)

    defender_x = rng.uniform(40, 85, n_samples)
    defender_y = rng.uniform(5, 95, n_samples)

    passer_x = rng.uniform(30, 100, n_samples)
    passer_y = rng.uniform(5, 95, n_samples)

    teammate_x = rng.uniform(20, 100, n_samples)
    teammate_y = rng.uniform(5, 95, n_samples)

    x_diff = teammate_x - defender_x

    label = (x_diff > 0).astype(int)

    # Add more samples near the decision boundary (x_diff close to 0)
    n_boundary = n_samples // 4
    defender_x_b = rng.uniform(40, 85, n_boundary)
    defender_y_b = rng.uniform(5, 95, n_boundary)
    passer_x_b = rng.uniform(30, 100, n_boundary)
    passer_y_b = rng.uniform(5, 95, n_boundary)
    # Generate x_diff values very close to 0 (between -0.5 and +0.5)
    x_diff_b = rng.uniform(-0.5, 0.5, n_boundary)
    teammate_x_b = defender_x_b + x_diff_b
    teammate_y_b = rng.uniform(5, 95, n_boundary)
    label_b = (x_diff_b > 0).astype(int)

    n_exact = n_samples // 8
    defender_x_e = rng.uniform(40, 85, n_exact)
    defender_y_e = rng.uniform(5, 95, n_exact)
    passer_x_e = rng.uniform(30, 100, n_exact)
    passer_y_e = rng.uniform(5, 95, n_exact)
    x_diff_e = np.zeros(n_exact)  # Exactly 0
    teammate_x_e = defender_x_e + x_diff_e
    teammate_y_e = rng.uniform(5, 95, n_exact)
    label_e = np.zeros(n_exact)  # All onside

    defender_x = np.concatenate([defender_x, defender_x_b, defender_x_e])
    defender_y = np.concatenate([defender_y, defender_y_b, defender_y_e])
    passer_x = np.concatenate([passer_x, passer_x_b, passer_x_e])
    passer_y = np.concatenate([passer_y, passer_y_b, passer_y_e])
    teammate_x = np.concatenate([teammate_x, teammate_x_b, teammate_x_e])
    teammate_y = np.concatenate([teammate_y, teammate_y_b, teammate_y_e])
    x_diff = np.concatenate([x_diff, x_diff_b, x_diff_e])
    label = np.concatenate([label, label_b, label_e])

    df = pd.DataFrame({
        "passer_x": passer_x,
        "passer_y": passer_y,
        "teammate_x": teammate_x,
        "teammate_y": teammate_y,
        "defender_x": defender_x,
        "defender_y": defender_y,
        "x_diff": x_diff,
    })
    df["offside"] = label

    return df


def generate_realtime_sample(passer_pos: tuple, teammate_pos: tuple, defender_pos: tuple) -> dict:
    px, py = passer_pos
    tx, ty = teammate_pos
    dx, dy = defender_pos
    return {
        "passer_x": [px],
        "passer_y": [py],
        "teammate_x": [tx],
        "teammate_y": [ty],
        "defender_x": [dx],
        "defender_y": [dy],
        "x_diff": [tx - dx],
    }


if __name__ == "__main__":
    df = generate_offside_sample(n_samples=100)
    print("=== Generated Sample Data ===")
    print(df.head(10).to_string(index=False))
    print(f"\nTotal: {len(df)} samples | Offside: {df['offside'].sum()} | Onside: {(df['offside'] == 0).sum()}")