"""Post-processing analyses for Spatial 3DCNN P-scores."""

from .pscore_analysis import (
    normalize_binary_scores,
    combine_cube_scores,
    cube_to_region_scores,
    region_scores_from_cubes,
    normalize_region_scores_minmax,
    region_summary_mean_std,
    degeneration_regions_from_threshold,
    region_frequency,
    total_brain_score,
)
from .connectivity_analysis import (
    build_region_adjacency,
    connected_components,
    spatial_components,
    spatiotemporal_components,
)
from .progression_analysis import (
    is_non_decreasing,
    final_increase,
    new_regions_over_time,
    continuous_growth_statistics,
    total_score_by_mmse,
)

__all__ = [name for name in globals() if not name.startswith("_")]
