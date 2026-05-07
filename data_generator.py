"""
SAOT - Semi-Automated Offside Technology
Module: Training Data Generator

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


def generate_offside_sample(n_samples: int = 2000, noise: float = 0.5, seed: int = 42):
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

    # Defender position: somewhere in opponent's half (40-85)
    defender_x = rng.uniform(40, 85, n_samples)
    defender_y = rng.uniform(5, 95, n_samples)

    # Passer position: the forward who makes the pass
    # Usually in advanced position but can be anywhere
    passer_x = rng.uniform(30, 100, n_samples)
    passer_y = rng.uniform(5, 95, n_samples)

    # Teammate position: the player receiving the pass
    # Can be anywhere on the field, with tendency towards offside positions
    teammate_x = rng.uniform(20, 100, n_samples)
    teammate_y = rng.uniform(5, 95, n_samples)

    # Label real (without noise): offside if teammate is ahead of defender
    x_diff = teammate_x - defender_x
    label_clean = (x_diff > 0).astype(int)

    # Add realistic noise (borderline cases ~noise%)
    flip_mask = rng.random(n_samples) < noise * 0.05  # ~2.5% flip rate
    label = label_clean.copy()
    label[flip_mask] = 1 - label[flip_mask]

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
    """
    Create a sample from a real-time reading (e.g., future OpenCV integration).

    Args:
        passer_pos: (x, y) coordinates of the passer (forward)
        teammate_pos: (x, y) coordinates of the teammate (co-player)
        defender_pos: (x, y) coordinates of the last defender

    Returns:
        dict with features ready for prediction
    """
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