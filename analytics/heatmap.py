"""
Heatmap Generation Module.

Builds 2D heatmaps from violation location data to visualize
areas with frequent violations. Helps traffic authorities
identify enforcement hotspots.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from typing import List, Dict, Optional

from utils.config import HEATMAP_WIDTH, HEATMAP_HEIGHT, HEATMAP_RADIUS, HEATMAP_DIR


def generate_heatmap(
    locations: List[Dict],
    output_path: Optional[str] = None,
    width: int = HEATMAP_WIDTH,
    height: int = HEATMAP_HEIGHT,
    radius: int = HEATMAP_RADIUS,
) -> str:
    """
    Generate a violation heatmap from location data.

    Algorithm:
        1. Initialize a zero matrix of (height, width).
        2. For each violation, increment the pixel at (x, y).
        3. Apply Gaussian blur to spread intensity.
        4. Render with a colormap (jet) and save as PNG.

    Args:
        locations: List of dicts with 'x', 'y' keys (pixel coordinates).
        output_path: Where to save the heatmap image.
        width: Heatmap image width.
        height: Heatmap image height.
        radius: Gaussian kernel radius for smoothing.

    Returns:
        Path to the saved heatmap image.
    """
    if output_path is None:
        output_path = os.path.join(str(HEATMAP_DIR), "violation_heatmap.png")

    # Initialize heatmap matrix
    heatmap = np.zeros((height, width), dtype=np.float32)

    # Accumulate violation positions
    for loc in locations:
        x = int(loc.get("x", 0))
        y = int(loc.get("y", 0))
        if 0 <= x < width and 0 <= y < height:
            heatmap[y, x] += 1

    # Apply Gaussian blur for smooth visualization
    if np.max(heatmap) > 0:
        heatmap = gaussian_filter(heatmap, sigma=radius)
        # Normalize to [0, 1]
        heatmap = heatmap / np.max(heatmap)

    # ── Render with matplotlib ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    im = ax.imshow(
        heatmap,
        cmap="jet",
        interpolation="bilinear",
        aspect="auto",
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        "Traffic Violation Heatmap",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Violation Intensity", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def generate_violation_type_heatmaps(
    locations: List[Dict],
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate separate heatmaps for each violation type.

    Returns:
        Dict mapping violation type → file path.
    """
    if output_dir is None:
        output_dir = str(HEATMAP_DIR)

    # Group by type
    by_type: Dict[str, list] = {}
    for loc in locations:
        vtype = loc.get("type", "unknown")
        by_type.setdefault(vtype, []).append(loc)

    paths = {}
    for vtype, locs in by_type.items():
        path = os.path.join(output_dir, f"heatmap_{vtype}.png")
        generate_heatmap(locs, output_path=path)
        paths[vtype] = path

    return paths
