"""Diagnostic surface quadrature fit; not used by the frozen flow validators.

Keep the physical probes, neighborhoods and tangential quadratic basis. Weight
each residual by face area times the existing distance kernel, so a small face
does not represent the same surface measure as a large one. No face is removed.
"""
import numpy as np
from scipy.spatial import cKDTree

from analyze_twisted_refinement import periodic_neighbors


def sample_wall_area_weighted(centers, values, areas, points, period, count,
                              normals, tree=None):
    centers = np.asarray(centers)
    values = np.asarray(values)
    areas = np.asarray(areas)
    if (areas.shape != (len(centers),) or not np.all(np.isfinite(areas)) or
            np.any(areas <= 0) or len(values) != len(centers)):
        raise ValueError('Require one positive finite area per surface sample')
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite surface field')
    tree = cKDTree(centers) if tree is None else tree
    ids, offsets = periodic_neighbors(tree, centers, points, period, count)
    result, conditions = [], []
    for i, (index, delta) in enumerate(zip(ids, offsets)):
        scale = np.max(np.linalg.norm(delta, axis=1))
        if scale <= 0:
            raise ValueError('Degenerate probe neighborhood')
        local = delta / scale
        normal = normals[i]
        axis = np.eye(3)[np.argmin(abs(normal))]
        tangent = np.cross(normal, axis)
        tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(normal, tangent)
        x, y = local @ tangent, local @ bitangent
        design = np.column_stack((np.ones(len(x)), x, y, x*x, x*y, y*y))
        # The common area normalization affects conditioning, not the minimizer.
        weight = np.sqrt((areas[index] / np.mean(areas[index])) /
                         (.05 + np.sum(local*local, axis=1)))
        rhs = values[index] * (weight[:, None] if values.ndim == 2 else weight)
        coef, _, rank, singular = np.linalg.lstsq(
            design * weight[:, None], rhs, rcond=1e-12)
        if rank != design.shape[1]:
            raise ValueError('Rank-deficient area-weighted surface fit')
        result.append(coef[0])
        conditions.append(float(singular[0] / singular[-1]))
    return np.asarray(result), max(conditions)
