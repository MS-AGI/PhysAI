"""
physai/geometry.py

Arbitrary-geometry support for PhysAI, built on signed distance functions
(SDFs). This is the same technique used under the hood by most modern
PINN toolkits: every shape — primitive or user-defined — is represented
as a function

    d(x) -> float

with d(x) < 0 inside the domain, d(x) == 0 on the boundary, and
d(x) > 0 outside. Complex shapes are built by combining primitive SDFs
with CSG operations (union, intersection, difference, smooth blends),
by writing a custom SDF directly in code, or by loading a surface mesh
from a file (.stl / .obj / .ply / .off).

Sections
--------
1.  Geometry base class          — common sampling/query API
2.  Primitives                   — box, ball, cylinder, half-space,
                                    ellipsoid, torus, capsule, polygon (2D)
3.  Transforms                   — translate, rotate, scale
4.  CSG combinators               — union, intersection, difference,
                                    smooth_union, invert
5.  User-defined geometry          — wrap any Python callable as an SDF
6.  File-based geometry            — load .stl/.obj/.ply/.off (optional
                                    trimesh dependency; numpy fallback)
7.  Boundary conditions            — attach Dirichlet/Neumann/Robin/
                                    periodic conditions to regions of a
                                    geometry's boundary
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_logger = logging.getLogger("physai")

ArrayFn = Callable[[np.ndarray], np.ndarray]


# ============================================================================
# 1. Geometry base class
# ============================================================================

class Geometry:
    """
    Wraps a signed distance function ``sdf(x) -> (n,)`` together with an
    axis-aligned bounding box, and provides the sampling / query utilities
    that the rest of PhysAI (samplers, trainer, boundary conditions) needs.

    Convention: sdf(x) < 0 inside the domain, > 0 outside, == 0 on ∂Ω.
    This is the standard SDF sign convention (note: this is the *opposite*
    sign of the older ``rectangular_distance_fn`` / ``spherical_distance_fn``
    helpers in ``utils.py``, which returned positive-inside "bump" functions
    for hard-constraint multiplication. Use ``.hard_constraint_fn()`` below
    to get a bump-style function compatible with those.)

    Parameters
    ----------
    sdf    : callable mapping (n, d) points -> (n,) signed distances
    bounds : axis-aligned bounding box, [(lo, hi), ...] per dimension —
             used to seed rejection sampling. Must fully contain the shape.
    dim    : spatial dimension (inferred from ``bounds`` if not given)
    name   : optional human-readable name, propagated through CSG ops
    """

    def __init__(
        self,
        sdf: ArrayFn,
        bounds: List[Tuple[float, float]],
        dim: Optional[int] = None,
        name: str = "geometry",
    ) -> None:
        self.sdf = sdf
        self.bounds = [(float(lo), float(hi)) for lo, hi in bounds]
        self.dim = dim if dim is not None else len(bounds)
        self.name = name

    # -- core queries --------------------------------------------------

    def distance(self, x: np.ndarray) -> np.ndarray:
        """Signed distance at points ``x`` of shape (n, d)."""
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        return np.asarray(self.sdf(x), dtype=np.float64)

    def contains(self, x: np.ndarray, tol: float = 0.0) -> np.ndarray:
        """Boolean mask: True where points are inside (or on, within tol)."""
        return self.distance(x) <= tol

    def hard_constraint_fn(self) -> ArrayFn:
        """
        Return a positive-inside "bump" function ``B(x)``, ``B=0`` on ∂Ω,
        ``B>0`` strictly inside — compatible with the hard-Dirichlet-constraint
        convention used by ``rectangular_distance_fn`` / ``spherical_distance_fn``
        in ``utils.py`` (multiply the network output by ``B(x)`` so boundary
        values vanish automatically).
        """
        def _bump(x: np.ndarray) -> np.ndarray:
            return -self.distance(x)
        return _bump

    # -- sampling --------------------------------------------------------

    def sample_interior(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        method: str = "uniform",
        oversample: float = 4.0,
        max_tries: int = 200,
    ) -> np.ndarray:
        """
        Rejection-sample ``n`` points strictly inside the domain.

        Draws candidate points from the bounding box (uniform or LHS) and
        keeps those with negative signed distance. Works for *any* shape,
        including CSG combinations and mesh-based geometries, at the cost
        of wasted samples for shapes that occupy a small fraction of their
        bounding box (e.g. a thin torus) — ``oversample`` controls the
        batch size multiplier per attempt.
        """
        from . import utils as _utils  # local import to avoid a cycle

        rng = rng or np.random.default_rng()
        collected: List[np.ndarray] = []
        got = 0
        tries = 0
        batch = max(int(n * oversample), n)

        while got < n and tries < max_tries:
            if method == "lhs":
                cand = _utils.sample_lhs(self.bounds, batch, rng=rng)
            else:
                cand = _utils.sample_uniform(self.bounds, batch, rng=rng)
            mask = self.distance(cand) < 0.0
            hits = cand[mask]
            if len(hits):
                collected.append(hits)
                got += len(hits)
            tries += 1
            batch = max(int(batch * 1.5), n)  # grow if acceptance rate is low

        if got < n:
            _logger.warning(
                f"sample_interior: only found {got}/{n} interior points for "
                f"'{self.name}' after {tries} attempts. The shape may occupy "
                f"a small fraction of its bounding box; consider tightening "
                f"``bounds`` or lowering ``n``."
            )

        if not collected:
            return np.empty((0, self.dim), dtype=np.float32)
        out = np.concatenate(collected, axis=0)[:n]
        return out.astype(np.float32)

    def sample_boundary(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
        shell: Optional[float] = None,
        max_tries: int = 400,
    ) -> np.ndarray:
        """
        Sample ``n`` points near ∂Ω via rejection sampling within a thin
        shell around the zero level-set, then projected onto the surface
        by one step of gradient (normal) descent on |sdf|.

        ``shell`` defaults to ~1% of the bounding-box diagonal.

        If this geometry has an exact analytic sampler attached (box,
        ball — see ``_exact_boundary_sampler``), that is used instead of
        shell-rejection: it's both cheaper and exact, and shell-rejection
        specifically struggles on thin/high-curvature shapes where the
        boundary shell occupies a tiny fraction of the bounding box.
        """
        if self._exact_boundary_sampler is not None:
            rng = rng or np.random.default_rng()
            return self._exact_boundary_sampler(n, rng)

        from . import utils as _utils

        rng = rng or np.random.default_rng()
        diag = float(np.linalg.norm(
            [hi - lo for lo, hi in self.bounds]
        ))
        shell = shell if shell is not None else max(diag * 0.01, 1e-4)

        collected: List[np.ndarray] = []
        got = 0
        tries = 0
        batch = max(n * 8, 256)

        while got < n and tries < max_tries:
            cand = _utils.sample_uniform(self.bounds, batch, rng=rng)
            d = self.distance(cand)
            mask = np.abs(d) < shell
            hits = cand[mask]
            if len(hits):
                hits = self._project_to_surface(hits)
                collected.append(hits)
                got += len(hits)
            tries += 1
            batch = min(int(batch * 1.5), 200_000)

        if got < n:
            _logger.warning(
                f"sample_boundary: only found {got}/{n} boundary points for "
                f"'{self.name}' after {tries} attempts."
            )

        if not collected:
            return np.empty((0, self.dim), dtype=np.float32)
        out = np.concatenate(collected, axis=0)[:n]
        return out.astype(np.float32)

    def _project_to_surface(self, x: np.ndarray, eps: float = 1e-5, steps: int = 3) -> np.ndarray:
        """Newton-style projection of points onto the d(x)=0 level set using a
        finite-difference gradient of the SDF (works for any SDF, including
        CSG combinations and mesh-based ones)."""
        pts = x.copy()
        for _ in range(steps):
            d = self.distance(pts)
            grad = self._numeric_gradient(pts, eps)
            gnorm2 = np.sum(grad ** 2, axis=-1) + 1e-12
            step = (d / gnorm2)[:, None] * grad
            pts = pts - step
        return pts

    def _numeric_gradient(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Central-difference gradient of the SDF at points x, shape (n, d)."""
        grad = np.zeros_like(x)
        for i in range(self.dim):
            dx = np.zeros(self.dim)
            dx[i] = eps
            grad[:, i] = (self.distance(x + dx) - self.distance(x - dx)) / (2 * eps)
        return grad

    def normal(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Outward unit normal at (near-boundary) points x, via SDF gradient."""
        grad = self._numeric_gradient(np.atleast_2d(x), eps)
        norm = np.linalg.norm(grad, axis=-1, keepdims=True) + 1e-12
        return grad / norm

    def project_to_boundary(self, x: np.ndarray, eps: float = 1e-5, steps: int = 3) -> np.ndarray:
        """
        Public entry point for the Newton-style SDF-gradient projection
        used internally by ``sample_boundary`` — maps arbitrary points
        (interior, exterior, or already near the boundary) onto ∂Ω. Works
        for any geometry (primitives, CSG combinations, meshes, point
        clouds), since it only needs ``self.distance``/its numeric
        gradient, not an analytic parametrization of the surface.

        Used by ``Solver.solve_geometry`` (see ``physai.solvers.solver``)
        to place its embedded-boundary substitution at the true surface
        location rather than at a raw exterior grid-neighbor coordinate.
        """
        return self._project_to_surface(np.atleast_2d(x), eps=eps, steps=steps)

    # -- exact analytic boundary sampling (primitives only) --------------
    # Set by primitive constructors (box, ball) that have a cheap, exact
    # parametric boundary sampler; `sample_boundary` prefers this over the
    # generic shell-rejection method when present, since shell-rejection
    # degrades badly for thin or high-curvature shapes (most surface area
    # for a very thin shell around ∂Ω is wasted rejecting interior/exterior
    # candidates). CSG combinations and mesh/point-cloud geometries have no
    # such hook and always use the generic method — composing an exact
    # sampler for an arbitrary CSG tree is a much larger undertaking than
    # this library attempts, and silently reusing one primitive's exact
    # sampler for a combined shape would be wrong.
    _exact_boundary_sampler: Optional[Callable[[int, np.random.Generator], np.ndarray]] = None

    # -- CSG operator overloads ------------------------------------------

    def __or__(self, other: "Geometry") -> "Geometry":
        return union(self, other)

    def __and__(self, other: "Geometry") -> "Geometry":
        return intersection(self, other)

    def __sub__(self, other: "Geometry") -> "Geometry":
        return difference(self, other)

    def __invert__(self) -> "Geometry":
        return invert(self)

    def __repr__(self) -> str:
        return f"Geometry(name={self.name!r}, dim={self.dim}, bounds={self.bounds})"


# ============================================================================
# 2. Primitives
# ============================================================================

def box(bounds: List[Tuple[float, float]], name: str = "box") -> Geometry:
    """Axis-aligned hyper-rectangle (exact SDF, any dimension)."""
    b = np.array(bounds, dtype=np.float64)
    lo, hi = b[:, 0], b[:, 1]
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0

    def _sdf(x: np.ndarray) -> np.ndarray:
        q = np.abs(x - center) - half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside

    geo = Geometry(_sdf, bounds, name=name)

    def _exact_sample(n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform-area sampling over an axis-aligned box's faces: pick a
        face weighted by its (d-1)-area, then sample uniformly on it."""
        dim = len(half)
        areas = np.array([
            np.prod([2 * half[j] for j in range(dim) if j != i]) for i in range(dim)
        ])
        face_probs = np.repeat(areas, 2)
        face_probs = face_probs / face_probs.sum()
        face_ids = rng.choice(2 * dim, size=n, p=face_probs)
        out = rng.uniform(-1.0, 1.0, size=(n, dim)) * half + center
        axis_ids = face_ids // 2
        signs = np.where(face_ids % 2 == 0, -1.0, 1.0)
        out[np.arange(n), axis_ids] = center[axis_ids] + signs * half[axis_ids]
        return out.astype(np.float32)

    geo._exact_boundary_sampler = _exact_sample
    return geo


def ball(center: Sequence[float], radius: float, name: str = "ball") -> Geometry:
    """d-dimensional ball (exact SDF)."""
    c = np.array(center, dtype=np.float64)
    d = len(c)

    def _sdf(x: np.ndarray) -> np.ndarray:
        return np.linalg.norm(x - c, axis=-1) - radius

    bounds = [(c[i] - radius, c[i] + radius) for i in range(d)]
    geo = Geometry(_sdf, bounds, dim=d, name=name)

    def _exact_sample(n: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform sampling on a (d-1)-sphere: normalize a Gaussian vector
        — the standard technique, exact in any dimension (unlike e.g.
        naive angle-based sampling, which biases toward the poles in 3D+)."""
        g = rng.normal(size=(n, d))
        g /= np.linalg.norm(g, axis=-1, keepdims=True) + 1e-300
        return (c + radius * g).astype(np.float32)

    geo._exact_boundary_sampler = _exact_sample
    return geo


def ellipsoid(center: Sequence[float], semi_axes: Sequence[float], name: str = "ellipsoid") -> Geometry:
    """
    Axis-aligned ellipsoid. Uses the standard (approximate, not exact)
    ellipsoid SDF, which is exact on the surface and a good first-order
    approximation nearby — fine for sampling and soft/hard BC use.
    """
    c = np.array(center, dtype=np.float64)
    a = np.array(semi_axes, dtype=np.float64)
    d = len(c)

    def _sdf(x: np.ndarray) -> np.ndarray:
        p = (x - c) / a
        k0 = np.linalg.norm(p, axis=-1)
        k1 = np.linalg.norm(p / a, axis=-1) + 1e-12
        return k0 * (k0 - 1.0) / k1

    bounds = [(c[i] - a[i], c[i] + a[i]) for i in range(d)]
    return Geometry(_sdf, bounds, dim=d, name=name)


def cylinder(
    point_a: Sequence[float],
    point_b: Sequence[float],
    radius: float,
    name: str = "cylinder",
) -> Geometry:
    """Finite 3D cylinder (capped) between two points, given radius."""
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    axis = b - a
    h = np.linalg.norm(axis)
    axis_n = axis / (h + 1e-12)

    def _sdf(x: np.ndarray) -> np.ndarray:
        pa = x - a
        proj = pa @ axis_n
        perp = pa - proj[:, None] * axis_n
        d_radial = np.linalg.norm(perp, axis=-1) - radius
        d_axial = np.abs(proj - h / 2.0) - h / 2.0
        outside = np.sqrt(np.maximum(d_radial, 0.0) ** 2 + np.maximum(d_axial, 0.0) ** 2)
        inside = np.minimum(np.maximum(d_radial, d_axial), 0.0)
        return outside + inside

    lo = np.minimum(a, b) - radius
    hi = np.maximum(a, b) + radius
    bounds = list(zip(lo.tolist(), hi.tolist()))
    return Geometry(_sdf, bounds, dim=3, name=name)


def torus(center: Sequence[float], major_radius: float, minor_radius: float, name: str = "torus") -> Geometry:
    """3D torus centered at ``center``, lying in the xy-plane (axis = z)."""
    c = np.array(center, dtype=np.float64)
    R, r = major_radius, minor_radius

    def _sdf(x: np.ndarray) -> np.ndarray:
        p = x - c
        q = np.stack([np.linalg.norm(p[:, :2], axis=-1) - R, p[:, 2]], axis=-1)
        return np.linalg.norm(q, axis=-1) - r

    ext = R + r
    bounds = [(c[0] - ext, c[0] + ext), (c[1] - ext, c[1] + ext), (c[2] - r, c[2] + r)]
    return Geometry(_sdf, bounds, dim=3, name=name)


def capsule(point_a: Sequence[float], point_b: Sequence[float], radius: float, name: str = "capsule") -> Geometry:
    """Line segment thickened by ``radius`` (2D or 3D)."""
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    d = len(a)
    ab = b - a
    ab2 = float(ab @ ab) + 1e-12

    def _sdf(x: np.ndarray) -> np.ndarray:
        ap = x - a
        t = np.clip((ap @ ab) / ab2, 0.0, 1.0)
        closest = a + t[:, None] * ab
        return np.linalg.norm(x - closest, axis=-1) - radius

    lo = np.minimum(a, b) - radius
    hi = np.maximum(a, b) + radius
    bounds = list(zip(lo.tolist(), hi.tolist()))
    return Geometry(_sdf, bounds, dim=d, name=name)


def half_space(point: Sequence[float], normal: Sequence[float], extent: float = 10.0, name: str = "half_space") -> Geometry:
    """
    Half-space {x : (x - point) . normal <= 0}, e.g. for domains cut by a
    plane. ``extent`` sets a finite bounding box (half-spaces are unbounded
    by nature, so this only matters for sampling/visualisation).
    """
    p = np.array(point, dtype=np.float64)
    n = np.array(normal, dtype=np.float64)
    n = n / (np.linalg.norm(n) + 1e-12)
    d = len(p)

    def _sdf(x: np.ndarray) -> np.ndarray:
        return (x - p) @ n

    bounds = [(p[i] - extent, p[i] + extent) for i in range(d)]
    return Geometry(_sdf, bounds, dim=d, name=name)


def polygon(vertices: np.ndarray, name: str = "polygon") -> Geometry:
    """
    Exact SDF for an arbitrary (possibly non-convex) simple 2D polygon,
    given its vertices in order. Sign is determined via the standard
    winding-number test.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape[-1] != 2:
        raise ValueError("polygon() expects 2D vertices of shape (m, 2).")
    n_v = len(v)

    def _sdf(x: np.ndarray) -> np.ndarray:
        n = len(x)
        dmin2 = np.full(n, np.inf)
        wind = np.zeros(n, dtype=np.int32)
        for i in range(n_v):
            v0 = v[i]
            v1 = v[(i + 1) % n_v]
            e = v1 - v0
            w = x - v0
            t = np.clip((w @ e) / (e @ e + 1e-12), 0.0, 1.0)
            proj = v0 + t[:, None] * e
            d2 = np.sum((x - proj) ** 2, axis=-1)
            dmin2 = np.minimum(dmin2, d2)

            cond1 = (v0[1] <= x[:, 1]) & (x[:, 1] < v1[1])
            cond2 = (v1[1] <= x[:, 1]) & (x[:, 1] < v0[1])
            cross = (v1[0] - v0[0]) * (x[:, 1] - v0[1]) - (v1[1] - v0[1]) * (x[:, 0] - v0[0])
            wind += np.where(cond1 & (cross > 0), 1, 0)
            wind -= np.where(cond2 & (cross < 0), 1, 0)

        sign = np.where(wind != 0, -1.0, 1.0)
        return sign * np.sqrt(dmin2)

    lo = v.min(axis=0)
    hi = v.max(axis=0)
    bounds = list(zip(lo.tolist(), hi.tolist()))
    return Geometry(_sdf, bounds, dim=2, name=name)


# ============================================================================
# 2b. Solids of revolution, extrusion, and periodic tiling
# ============================================================================
# These build 3D (or higher) geometry by composing an already-exact lower-
# dimensional Geometry (typically `polygon` or `box`) rather than deriving
# new closed-form SDFs by hand — composition inherits whatever exactness
# guarantee the inner geometry already has, instead of risking a fresh,
# unverified analytic formula. See revolve()'s docstring for a correctness
# subtlety (the profile must be the *mirrored* full cross-section, not the
# bare half-profile) that a naive implementation of this trick gets wrong.

def _orthonormal_basis(axis_n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Any two unit vectors perpendicular to ``axis_n`` and to each other,
    for building a local 2D frame transverse to a 3D axis."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis_n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis_n, ref)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(axis_n, u)
    return u, v


def revolve(profile: Geometry, point_a: Sequence[float], point_b: Sequence[float], name: str = "revolve") -> Geometry:
    """
    3D solid of revolution: sweep a 2D cross-section ``profile`` (defined
    in a half-plane with its first coordinate = radius ``r >= 0`` and
    second coordinate = height ``t`` along the axis) around the axis from
    ``point_a`` to ``point_b``.

    IMPORTANT: ``profile`` must be built as the *mirrored, full* 2D shape
    — symmetric about ``r = 0`` — not just the ``r >= 0`` half. E.g. for a
    cylinder of radius R and height h, the correct profile is
    ``polygon([(-R, 0), (R, 0), (R, h), (-R, h)])``, not
    ``polygon([(0, 0), (R, 0), (R, h), (0, h)])``. The half-profile's edge
    running along ``r = 0`` is a genuine polygon edge — the revolve query
    (which only ever evaluates at ``r = sqrt(x^2+z^2) >= 0``) would then
    read points on the rotation axis as lying *on the boundary* rather
    than deep in the interior, corrupting both the sign and magnitude of
    the resulting 3D SDF near the axis (verified numerically against
    ``cylinder()`` while developing this function: the half-profile
    version was wrong by O(R) near the axis; the mirrored version matches
    ``cylinder()`` to float64 precision).

    Convenience wrappers ``cone``/``frustum`` below build the right
    mirrored profile automatically — reach for those unless you need an
    arbitrary revolved cross-section.
    """
    if profile.dim != 2:
        raise ValueError("revolve: profile must be a 2D Geometry.")
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    axis = b - a
    h = float(np.linalg.norm(axis))
    axis_n = axis / (h + 1e-12)

    def _sdf(x: np.ndarray) -> np.ndarray:
        rel = x - a
        t = rel @ axis_n
        perp = rel - t[:, None] * axis_n
        r = np.linalg.norm(perp, axis=-1)
        return profile.sdf(np.stack([r, t], axis=-1))

    r_max = float(np.max(np.abs(np.asarray(profile.bounds))))
    lo3 = np.minimum(a, b) - r_max
    hi3 = np.maximum(a, b) + r_max
    bounds = list(zip(lo3.tolist(), hi3.tolist()))
    return Geometry(_sdf, bounds, dim=3, name=name)


def extrude(profile: Geometry, point_a: Sequence[float], point_b: Sequence[float], name: str = "extrude") -> Geometry:
    """
    3D prism: sweep a 2D cross-section ``profile`` (its own (x, y) plane)
    along the straight axis from ``point_a`` to ``point_b`` — the profile
    is placed in the plane perpendicular to that axis at every point along
    it, via an arbitrary (but consistent) orthonormal frame transverse to
    the axis (``_orthonormal_basis``). Verified against ``box()`` for an
    axis-aligned square profile extruded along z (exact match to float
    precision); works for any orientation of ``point_a``/``point_b``, not
    just axis-aligned, since the transverse frame is rebuilt from the
    actual axis direction.
    """
    if profile.dim != 2:
        raise ValueError("extrude: profile must be a 2D Geometry.")
    a = np.array(point_a, dtype=np.float64)
    b = np.array(point_b, dtype=np.float64)
    axis = b - a
    h = float(np.linalg.norm(axis))
    axis_n = axis / (h + 1e-12)
    u, v = _orthonormal_basis(axis_n)

    def _sdf(x: np.ndarray) -> np.ndarray:
        rel = x - a
        t = rel @ axis_n
        pu = rel @ u
        pv = rel @ v
        d2 = profile.sdf(np.stack([pu, pv], axis=-1))
        dz = np.abs(t - h / 2.0) - h / 2.0
        outside = np.sqrt(np.maximum(d2, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
        inside = np.minimum(np.maximum(d2, dz), 0.0)
        return outside + inside

    prof_b = np.asarray(profile.bounds)
    r_max = float(np.max(np.abs(prof_b)))
    lo3 = np.minimum(a, b) - r_max
    hi3 = np.maximum(a, b) + r_max
    bounds = list(zip(lo3.tolist(), hi3.tolist()))
    return Geometry(_sdf, bounds, dim=3, name=name)


def cone(apex: Sequence[float], base: Sequence[float], radius: float, name: str = "cone") -> Geometry:
    """Solid circular cone: apex at ``apex`` (radius 0), base at ``base``
    (radius ``radius``). Built via ``revolve`` of an exact triangular
    profile — see ``revolve``'s docstring for why the profile must be the
    mirrored full triangle, not just the ``r >= 0`` half."""
    h = float(np.linalg.norm(np.array(base, dtype=np.float64) - np.array(apex, dtype=np.float64)))
    profile = polygon(np.array([[0.0, 0.0], [radius, h], [-radius, h]]))
    return revolve(profile, apex, base, name=name)


def frustum(point_a: Sequence[float], radius_a: float, point_b: Sequence[float], radius_b: float, name: str = "frustum") -> Geometry:
    """Solid truncated cone (frustum) between two circular faces of
    radius ``radius_a``/``radius_b``. Built via ``revolve`` of an exact
    trapezoidal profile."""
    h = float(np.linalg.norm(np.array(point_b, dtype=np.float64) - np.array(point_a, dtype=np.float64)))
    profile = polygon(np.array([
        [-radius_a, 0.0], [radius_a, 0.0], [radius_b, h], [-radius_b, h],
    ]))
    return revolve(profile, point_a, point_b, name=name)


def tile(geo: Geometry, period: Sequence[float], reps: Optional[Sequence[int]] = None, name: Optional[str] = None) -> Geometry:
    """
    Periodically repeat ``geo`` along each axis with the given ``period``.
    Standard infinite-domain-repetition SDF technique: fold query points
    into a single period cell (``mod(x + period/2, period) - period/2``)
    before evaluating the base SDF. Verified numerically: tile-center
    points at ``0, period, -period, ...`` all correctly evaluate as
    strongly inside.

    If ``reps`` (a per-axis integer count) is given, the repetition is
    clamped to that many copies per axis instead of being infinite (the
    standard finite-repetition variant: clamp the tile *index*, not the
    folded coordinate, to ``[0, reps-1]`` before folding) — this bounds
    ``bounds`` to a finite box; otherwise ``bounds`` is only meaningful
    locally (sampling still needs a finite box, so infinite tiling keeps
    the original geometry's own bounds, i.e. sampling only ever explores
    one period's worth by default — pass a wider explicit ``bounds`` via
    ``custom_geometry`` wrapping this SDF if a multi-period sampling
    region is actually needed).
    """
    period_arr = np.asarray(period, dtype=np.float64)
    dim = geo.dim

    if reps is not None:
        reps_arr = np.asarray(reps, dtype=np.float64)
        origin = np.array([lo for lo, hi in geo.bounds])

        def _sdf(x: np.ndarray) -> np.ndarray:
            idx = np.clip(np.round((x - origin) / period_arr), 0, reps_arr - 1)
            xm = x - idx * period_arr
            return geo.sdf(xm)

        lo = [geo.bounds[i][0] for i in range(dim)]
        hi = [geo.bounds[i][0] + reps_arr[i] * period_arr[i] + (geo.bounds[i][1] - geo.bounds[i][0]) for i in range(dim)]
        bounds = list(zip(lo, hi))
    else:
        def _sdf(x: np.ndarray) -> np.ndarray:
            xm = np.mod(x + 0.5 * period_arr, period_arr) - 0.5 * period_arr
            return geo.sdf(xm)

        bounds = geo.bounds

    return Geometry(_sdf, bounds, dim=dim, name=name or f"tile({geo.name})")


def rounded_box(bounds: List[Tuple[float, float]], radius: float, name: str = "rounded_box") -> Geometry:
    """
    Axis-aligned box with all edges/corners filleted to ``radius``. Exact
    SDF (the standard "shrink by radius, take exact-box distance, restore
    radius" construction) — verified against hand-derived analytic
    distances at a flat-face point and at a rounded-corner point while
    developing this function (both matched to float precision).
    """
    b = np.array(bounds, dtype=np.float64)
    lo, hi = b[:, 0], b[:, 1]
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    inner_half = half - radius
    if np.any(inner_half < 0):
        raise ValueError(
            f"rounded_box: radius={radius} is larger than half the box's "
            f"shortest side ({float(np.min(half)):.4g}) along at least one "
            f"axis — the fillet can't exceed the box's own half-extent."
        )

    def _sdf(x: np.ndarray) -> np.ndarray:
        q = np.abs(x - center) - inner_half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        inside = np.minimum(np.max(q, axis=-1), 0.0)
        return outside + inside - radius

    return Geometry(_sdf, bounds, name=name)


def round_geometry(geo: Geometry, radius: float, name: Optional[str] = None) -> Geometry:
    """
    Generic Minkowski rounding: fillets any geometry's boundary by
    ``radius`` (``d(x) -> d(x) - radius``). Exact whenever ``geo``'s own
    SDF is an exact Euclidean distance (box, ball, polygon, revolve/
    extrude of those, CSG of those); inherits the same first-order
    approximation ``ellipsoid``/``scale`` already carry if applied to one
    of those. Bounding box is expanded by ``radius`` on every side.
    """
    def _sdf(x: np.ndarray) -> np.ndarray:
        return geo.sdf(x) - radius

    bounds = [(lo - radius, hi + radius) for lo, hi in geo.bounds]
    return Geometry(_sdf, bounds, dim=geo.dim, name=name or f"round({geo.name}, {radius})")


def shell(geo: Geometry, thickness: float, name: Optional[str] = None) -> Geometry:
    """
    Turn a solid ``geo`` into a thin shell of the given ``thickness``,
    centered on its original boundary (``d(x) -> |d(x)| - thickness/2``).
    Useful for modeling walls, membranes, or thin-walled vessels from a
    solid CAD-style geometry rather than defining the shell directly.
    """
    half_t = thickness / 2.0

    def _sdf(x: np.ndarray) -> np.ndarray:
        return np.abs(geo.sdf(x)) - half_t

    bounds = [(lo - half_t, hi + half_t) for lo, hi in geo.bounds]
    return Geometry(_sdf, bounds, dim=geo.dim, name=name or f"shell({geo.name}, {thickness})")




def translate(geo: Geometry, offset: Sequence[float]) -> Geometry:
    off = np.array(offset, dtype=np.float64)

    def _sdf(x: np.ndarray) -> np.ndarray:
        return geo.sdf(x - off)

    bounds = [(lo + off[i], hi + off[i]) for i, (lo, hi) in enumerate(geo.bounds)]
    return Geometry(_sdf, bounds, dim=geo.dim, name=f"translate({geo.name})")


def scale(geo: Geometry, factor: Union[float, Sequence[float]]) -> Geometry:
    """Uniform (float) or per-axis (sequence) scaling about the origin."""
    f = np.asarray(factor, dtype=np.float64)
    is_uniform = f.ndim == 0

    def _sdf(x: np.ndarray) -> np.ndarray:
        x2 = x / f
        d = geo.sdf(x2)
        # For uniform scale the SDF scales exactly; for non-uniform scale
        # this is an approximation (common, standard practice), good enough
        # for sampling and soft constraints.
        return d * float(np.min(f)) if is_uniform else d * float(np.min(f))

    bounds = [(lo * float(f if is_uniform else f[i]), hi * float(f if is_uniform else f[i]))
              for i, (lo, hi) in enumerate(geo.bounds)]
    bounds = [(min(lo, hi), max(lo, hi)) for lo, hi in bounds]
    return Geometry(_sdf, bounds, dim=geo.dim, name=f"scale({geo.name})")


def rotate2d(geo: Geometry, angle_rad: float, center: Sequence[float] = (0.0, 0.0)) -> Geometry:
    """Rotate a 2D geometry by ``angle_rad`` about ``center``."""
    if geo.dim != 2:
        raise ValueError("rotate2d only applies to 2D geometries; see rotate3d for 3D.")
    c = np.array(center, dtype=np.float64)
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[ca, sa], [-sa, ca]])  # inverse rotation, applied to query points

    def _sdf(x: np.ndarray) -> np.ndarray:
        return geo.sdf((x - c) @ R.T + c)

    corners = np.array([[lo, hi] for lo, hi in geo.bounds])
    all_corners = np.array(np.meshgrid(*corners)).reshape(2, -1).T
    Rf = np.array([[ca, -sa], [sa, ca]])
    rotated = (all_corners - c) @ Rf.T + c
    lo = rotated.min(axis=0)
    hi = rotated.max(axis=0)
    bounds = list(zip(lo.tolist(), hi.tolist()))
    return Geometry(_sdf, bounds, dim=2, name=f"rotate({geo.name})")


def rotate3d(geo: Geometry, axis: Sequence[float], angle_rad: float, center: Sequence[float] = (0.0, 0.0, 0.0)) -> Geometry:
    """Rotate a 3D geometry by ``angle_rad`` about ``axis`` through ``center`` (Rodrigues' formula)."""
    if geo.dim != 3:
        raise ValueError("rotate3d only applies to 3D geometries; see rotate2d for 2D.")
    c = np.array(center, dtype=np.float64)
    a = np.array(axis, dtype=np.float64)
    a = a / (np.linalg.norm(a) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    Rf = np.eye(3) + sa * K + (1 - ca) * (K @ K)  # forward rotation
    Rinv = Rf.T  # rotation matrices are orthogonal

    def _sdf(x: np.ndarray) -> np.ndarray:
        return geo.sdf((x - c) @ Rinv.T + c)

    corners = np.array([[lo, hi] for lo, hi in geo.bounds])
    all_corners = np.array(np.meshgrid(*corners)).reshape(3, -1).T
    rotated = (all_corners - c) @ Rf.T + c
    lo = rotated.min(axis=0)
    hi = rotated.max(axis=0)
    bounds = list(zip(lo.tolist(), hi.tolist()))
    return Geometry(_sdf, bounds, dim=3, name=f"rotate({geo.name})")


# ============================================================================
# 4. CSG combinators
# ============================================================================

def _merged_bounds(geoms: Sequence[Geometry]) -> List[Tuple[float, float]]:
    dim = geoms[0].dim
    lo = [min(g.bounds[i][0] for g in geoms) for i in range(dim)]
    hi = [max(g.bounds[i][1] for g in geoms) for i in range(dim)]
    return list(zip(lo, hi))


def _intersected_bounds(geoms: Sequence[Geometry]) -> List[Tuple[float, float]]:
    dim = geoms[0].dim
    lo = [max(g.bounds[i][0] for g in geoms) for i in range(dim)]
    hi = [min(g.bounds[i][1] for g in geoms) for i in range(dim)]
    return list(zip(lo, hi))


def union(*geoms: Geometry) -> Geometry:
    """CSG union (A ∪ B ∪ ...): d = min(d_1, d_2, ...)."""
    if len(geoms) == 1:
        return geoms[0]

    def _sdf(x: np.ndarray) -> np.ndarray:
        return np.min(np.stack([g.sdf(x) for g in geoms], axis=0), axis=0)

    name = " | ".join(g.name for g in geoms)
    return Geometry(_sdf, _merged_bounds(geoms), dim=geoms[0].dim, name=f"({name})")


def intersection(*geoms: Geometry) -> Geometry:
    """CSG intersection (A ∩ B ∩ ...): d = max(d_1, d_2, ...)."""
    if len(geoms) == 1:
        return geoms[0]

    def _sdf(x: np.ndarray) -> np.ndarray:
        return np.max(np.stack([g.sdf(x) for g in geoms], axis=0), axis=0)

    name = " & ".join(g.name for g in geoms)
    return Geometry(_sdf, _intersected_bounds(geoms), dim=geoms[0].dim, name=f"({name})")


def difference(a: Geometry, b: Geometry) -> Geometry:
    """CSG subtraction (A \\ B): d = max(d_A, -d_B)."""
    def _sdf(x: np.ndarray) -> np.ndarray:
        return np.maximum(a.sdf(x), -b.sdf(x))

    return Geometry(_sdf, a.bounds, dim=a.dim, name=f"({a.name} - {b.name})")


def invert(geo: Geometry) -> Geometry:
    """Complement of a geometry (flips inside/outside)."""
    def _sdf(x: np.ndarray) -> np.ndarray:
        return -geo.sdf(x)

    return Geometry(_sdf, geo.bounds, dim=geo.dim, name=f"~{geo.name}")


def smooth_union(a: Geometry, b: Geometry, k: float = 0.1) -> Geometry:
    """
    Polynomial smooth-min union — blends two shapes with a smooth fillet
    of size ``k`` instead of a sharp CSG seam. Useful when a hard union
    would create a non-differentiable kink that hurts PINN training near
    the seam.
    """
    def _sdf(x: np.ndarray) -> np.ndarray:
        d1, d2 = a.sdf(x), b.sdf(x)
        h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
        return d2 * (1 - h) + d1 * h - k * h * (1 - h)

    return Geometry(_sdf, _merged_bounds([a, b]), dim=a.dim, name=f"smooth_union({a.name}, {b.name})")


def smooth_difference(a: Geometry, b: Geometry, k: float = 0.1) -> Geometry:
    """Smooth version of ``difference`` — rounds the cut edge instead of leaving a sharp seam."""
    def _sdf(x: np.ndarray) -> np.ndarray:
        d1, d2 = a.sdf(x), -b.sdf(x)
        h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
        return d2 * (1 - h) + d1 * h + k * h * (1 - h)

    return Geometry(_sdf, a.bounds, dim=a.dim, name=f"smooth_diff({a.name}, {b.name})")


def smooth_intersection(a: Geometry, b: Geometry, k: float = 0.1) -> Geometry:
    """Smooth version of ``intersection`` — polynomial smooth-max, same
    ``k``-sized fillet convention as ``smooth_union``/``smooth_difference``."""
    def _sdf(x: np.ndarray) -> np.ndarray:
        d1, d2 = a.sdf(x), b.sdf(x)
        h = np.clip(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0)
        return d2 * (1 - h) + d1 * h + k * h * (1 - h)

    return Geometry(_sdf, _intersected_bounds([a, b]), dim=a.dim, name=f"smooth_intersection({a.name}, {b.name})")


# ============================================================================
# 5. User-defined geometry
# ============================================================================

def custom_geometry(
    sdf_fn: ArrayFn,
    bounds: List[Tuple[float, float]],
    name: str = "custom",
    validate: bool = True,
    n_validate: int = 64,
) -> Geometry:
    """
    Wrap any user-written Python function as a PhysAI geometry, so the
    user can define arbitrary domains directly in code:

    >>> def my_sdf(x):
    ...     # a superellipse in 2D: |x|^4 + |y|^4 <= 1
    ...     return (np.abs(x[:, 0]) ** 4 + np.abs(x[:, 1]) ** 4) - 1.0
    >>> geo = custom_geometry(my_sdf, bounds=[(-1.2, 1.2), (-1.2, 1.2)])

    ``sdf_fn`` must accept an (n, d) array and return an (n,) array of
    signed distances (negative inside). It does *not* need to be an exact
    Euclidean distance — any function that is negative inside, positive
    outside, and zero on the boundary works correctly for sampling, CSG
    composition, and hard-constraint multiplication (only the exactness
    of distance *magnitude* — not the sign — matters for those uses;
    gradient-based normals rely on this being smooth, not on it being an
    exact distance).

    ``validate`` runs a quick smoke test (shape/dtype/NaN check) against
    random points in ``bounds`` so mistakes surface immediately instead
    of silently breaking sampling deep inside a training loop.
    """
    if validate:
        rng = np.random.default_rng(0)
        probe = np.stack(
            [rng.uniform(lo, hi, n_validate) for lo, hi in bounds], axis=1
        ).astype(np.float64)
        try:
            out = np.asarray(sdf_fn(probe))
        except Exception as exc:  # noqa: BLE001 - surfacing user's own bug clearly
            raise ValueError(
                f"custom_geometry: sdf_fn raised an exception when called on "
                f"a ({n_validate}, {len(bounds)}) probe array: {exc}"
            ) from exc
        if out.shape != (n_validate,):
            raise ValueError(
                f"custom_geometry: sdf_fn must return shape ({n_validate},) "
                f"for an ({n_validate}, {len(bounds)}) input; got {out.shape}. "
                f"Make sure it reduces over the last axis (e.g. use axis=-1)."
            )
        if not np.all(np.isfinite(out)):
            _logger.warning(
                "custom_geometry: sdf_fn produced non-finite values on the "
                "validation probe — check for divide-by-zero at the origin "
                "or similar singularities."
            )

    return Geometry(sdf_fn, bounds, dim=len(bounds), name=name)


# ============================================================================
# 5b. Point-cloud geometry
# ============================================================================

def point_cloud_geometry(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    name: str = "point_cloud",
) -> Geometry:
    """
    Build a Geometry directly from a raw point cloud — e.g. LIDAR/3D-scan
    data, or any surface sample the user already has without a meshed
    surface to go with it.

    Nearest-neighbor queries use ``scipy.spatial.cKDTree`` (O(log n) per
    query after an O(n log n) build), not brute-force pairwise distance —
    this matters in practice: brute force is O(n_query * n_points), which
    is the dominant cost for any nontrivial point cloud during training's
    repeated ``sample_interior``/``sample_boundary`` calls.

    Parameters
    ----------
    points  : (n, d) array of surface points.
    normals : optional (n, d) array of *outward* unit normals, one per
        point. If given, this produces a genuinely SIGNED distance: for a
        query point, take its nearest cloud point and use that point's
        normal to determine which side of the surface the query is on
        (``sign = sign((x - nearest) . normal)``) — the standard
        point-cloud signed-distance technique, exact in the limit of
        dense, correctly-oriented sampling and a good local approximation
        otherwise (verified numerically against a densely-sampled circle
        with known analytic normals while developing this function: exact
        sign, distance accurate to the point-cloud's own sampling
        resolution). If omitted, the distance is UNSIGNED (nearest-point
        distance only, always >= 0) — honestly making this geometry
        usable for boundary sampling and soft penalty losses, but NOT for
        ``.contains()`` or hard-constraint multiplication (same documented
        limitation as ``geometry_from_file``'s no-trimesh fallback, for
        the same underlying reason: there is no inside/outside information
        in an unoriented point cloud to recover).
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError(f"point_cloud_geometry: points must be (n, d), got shape {pts.shape}.")
    dim = pts.shape[1]

    if normals is None:
        _logger.warning(
            "point_cloud_geometry: no `normals` given, so the resulting "
            "geometry is UNSIGNED (nearest-point distance only). This is "
            "fine for boundary sampling and soft penalty losses, but "
            "`.contains()` / hard-constraint multiplication will not be "
            "correct — pass outward unit normals (e.g. estimated via PCA "
            "over local neighborhoods, or from your scanner's own output) "
            "for a true signed distance."
        )
    _sdf = _kdtree_sdf(pts, normals)

    lo = pts.min(axis=0).tolist()
    hi = pts.max(axis=0).tolist()
    bounds = list(zip(lo, hi))
    geo = Geometry(_sdf, bounds, dim=dim, name=name)
    geo._is_unsigned_fallback = normals is None
    geo._point_cloud = pts
    return geo


def _kdtree_sdf(points: np.ndarray, normals: Optional[np.ndarray]) -> ArrayFn:
    """
    Shared implementation behind ``point_cloud_geometry`` and
    ``geometry_from_file``'s no-trimesh fallback: an ``scipy.spatial.
    cKDTree``-backed nearest-point distance, signed via the nearest
    point's own normal when ``normals`` is given (see
    ``point_cloud_geometry``'s docstring for the technique and its
    numerical verification), unsigned otherwise.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(points)

    if normals is not None:
        n = np.asarray(normals, dtype=np.float64)
        if n.shape != points.shape:
            raise ValueError(f"_kdtree_sdf: normals shape {n.shape} must match points shape {points.shape}.")
        n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)

        def _sdf(x: np.ndarray) -> np.ndarray:
            dist, idx = tree.query(x)
            nearest = points[idx]
            sign = np.sign(np.sum((x - nearest) * n[idx], axis=-1))
            sign = np.where(sign == 0, 1.0, sign)
            return sign * dist
        return _sdf

    def _sdf_unsigned(x: np.ndarray) -> np.ndarray:
        dist, _ = tree.query(x)
        return dist
    return _sdf_unsigned


# ============================================================================
# 6. File-based geometry (user-uploaded meshes)
# ============================================================================

def geometry_from_file(
    path: Union[str, Path],
    name: Optional[str] = None,
    n_surface_samples: int = 20_000,
) -> Geometry:
    """
    Build a Geometry from a user-uploaded surface mesh file
    (.stl, .obj, .ply, .off, and anything else ``trimesh`` can load).

    Three backends, chosen automatically, in order of preference:

    * If ``trimesh`` is installed, this uses ``trimesh.proximity.signed_distance``,
      which gives a true signed distance for any watertight mesh (correct
      inside/outside via winding, not just nearest-point distance).
    * If ``trimesh`` is not installed but the file has per-face
      orientation information PhysAI's own lightweight parser can recover
      — STL always does (its format stores a facet normal for every
      triangle directly, unrelated to vertex winding order), and OBJ does
      whenever it has ``f`` (face) lines, via the standard
      cross-product-of-edges face normal (assumes consistent
      counter-clockwise winding, the standard OBJ convention — noted here
      since a malformed/inconsistently-wound OBJ would silently produce
      wrong signs, same caveat any face-normal-based method has) — this
      builds a genuinely SIGNED nearest-point distance from those face
      centroids/normals via ``_kdtree_sdf``, without needing trimesh at
      all.
    * Only if neither is available (e.g. a bare OBJ point cloud with no
      face data at all) does this fall back to the UNSIGNED nearest-
      vertex distance — suitable for boundary sampling and soft penalty
      losses only, same caveat as ``point_cloud_geometry``'s no-normals case.

    Parameters
    ----------
    path : path to the mesh file the user uploaded
    name : defaults to the file stem
    n_surface_samples : target point-cloud density for the face-normal /
        no-trimesh paths (points are distributed across faces
        proportional to face area, so large faces get proportionally more
        samples rather than one point regardless of size)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"geometry_from_file: no such file: {path}")
    name = name or path.stem

    try:
        import trimesh
    except ImportError:
        trimesh = None

    if trimesh is not None:
        mesh = trimesh.load(path, force="mesh")
        if not mesh.is_watertight:
            _logger.warning(
                f"geometry_from_file: '{path.name}' is not watertight — "
                f"signed distance (inside/outside) may be unreliable near "
                f"gaps in the mesh. Boundary sampling will still work."
            )

        def _sdf(x: np.ndarray) -> np.ndarray:
            # trimesh convention: positive = inside. Flip to PhysAI's
            # negative-inside convention.
            return -trimesh.proximity.signed_distance(mesh, x)

        lo = mesh.bounds[0].tolist()
        hi = mesh.bounds[1].tolist()
        bounds = list(zip(lo, hi))
        geo = Geometry(_sdf, bounds, dim=3, name=name)
        geo._mesh = mesh  # stashed for visualization / export use
        return geo

    verts, faces = _load_mesh_faces(path)

    if faces is not None:
        centroids, normals = _face_samples(verts, faces, n_surface_samples)
        _logger.info(
            f"geometry_from_file: 'trimesh' not installed, using PhysAI's "
            f"own face-normal-based signed distance for '{path.name}' "
            f"({len(faces)} faces -> {len(centroids)} oriented surface "
            f"samples). This is genuinely signed (not the unsigned "
            f"nearest-vertex fallback) — install `trimesh` only if you "
            f"need exact/watertight-verified signed distance instead."
        )
        _sdf = _kdtree_sdf(centroids, normals)
        lo = verts.min(axis=0).tolist()
        hi = verts.max(axis=0).tolist()
        bounds = list(zip(lo, hi))
        geo = Geometry(_sdf, bounds, dim=verts.shape[1], name=name)
        geo._is_unsigned_fallback = False
        return geo

    # ---- last-resort fallback: unsigned nearest-vertex distance ----------
    _logger.warning(
        f"geometry_from_file: '{path.name}' has no face/triangle data "
        f"this parser could extract (and 'trimesh' is not installed), so "
        f"mesh loading is using an UNSIGNED point-cloud fallback "
        f"(nearest-vertex distance only — no reliable inside/outside "
        f"test). This is fine for boundary point sampling and soft "
        f"losses, but `.contains()` and hard-constraint multiplication "
        f"will not be correct. Run `pip install trimesh` for exact "
        f"signed distance, or provide a file with face data."
    )
    _sdf_unsigned = _kdtree_sdf(verts, None)
    lo = verts.min(axis=0).tolist()
    hi = verts.max(axis=0).tolist()
    bounds = list(zip(lo, hi))
    geo = Geometry(_sdf_unsigned, bounds, dim=verts.shape[1], name=name)
    geo._is_unsigned_fallback = True
    return geo


def _face_samples(verts: np.ndarray, faces: np.ndarray, n_target: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Turn a triangle mesh into an oriented point cloud for ``_kdtree_sdf``:
    each face contributes its centroid plus (for larger faces) extra
    barycentric samples, distributed proportional to face area so a big
    flat face isn't under-represented relative to a cluster of tiny ones
    — every sample carries that face's own normal.
    """
    tri = verts[faces]  # (m, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=-1)
    normals = cross / (np.linalg.norm(cross, axis=-1, keepdims=True) + 1e-12)

    total_area = areas.sum() + 1e-12
    counts = np.maximum(1, np.round(n_target * areas / total_area).astype(int))

    rng = np.random.default_rng(0)
    all_pts = []
    all_normals = []
    for i in range(len(faces)):
        k = int(counts[i])
        r1 = rng.random(k)
        r2 = rng.random(k)
        sqrt_r1 = np.sqrt(r1)
        bary_a = 1 - sqrt_r1
        bary_b = sqrt_r1 * (1 - r2)
        bary_c = sqrt_r1 * r2
        pts = (
            bary_a[:, None] * tri[i, 0]
            + bary_b[:, None] * tri[i, 1]
            + bary_c[:, None] * tri[i, 2]
        )
        all_pts.append(pts)
        all_normals.append(np.tile(normals[i], (k, 1)))

    return np.concatenate(all_pts, axis=0), np.concatenate(all_normals, axis=0)


def _load_mesh_faces(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Dependency-free vertex + face extraction for the trimesh-less path.
    Returns ``(vertices, faces)`` — ``faces`` is ``None`` if the file has
    no face/triangle connectivity this parser can recover (e.g. an OBJ
    with only ``v`` lines), signaling the caller to fall back to unsigned
    nearest-vertex distance.

    STL always has face data (it's defined as a flat list of triangles;
    there's no such thing as an STL "point cloud"). OBJ has face data
    only if it has ``f`` lines.
    """
    suffix = path.suffix.lower()

    if suffix == ".stl":
        data = path.read_bytes()
        if data[:5].lower() == b"solid" and b"facet normal" in data[:1024]:
            verts = []
            for line in data.decode("ascii", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("vertex"):
                    verts.append([float(v) for v in line.split()[1:4]])
            if not verts:
                raise ValueError(f"No vertices found while parsing '{path.name}' as ASCII STL.")
            verts = np.array(verts, dtype=np.float64)
        else:
            n_tri = int(np.frombuffer(data[80:84], dtype=np.uint32)[0])
            verts = np.empty((n_tri * 3, 3), dtype=np.float32)
            offset = 84
            for i in range(n_tri):
                tri = np.frombuffer(data[offset + 12: offset + 48], dtype=np.float32).reshape(3, 3)
                verts[i * 3: i * 3 + 3] = tri
                offset += 50
            verts = verts.astype(np.float64)
        # STL triangles are already flat (3 verts per face, no sharing) —
        # faces are just consecutive triples.
        n_tri = len(verts) // 3
        faces = np.arange(n_tri * 3).reshape(n_tri, 3)
        return verts, faces

    if suffix == ".obj":
        verts = []
        faces = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if line.startswith("v "):
                    parts = line.split()[1:4]
                    verts.append([float(p) for p in parts])
                elif line.startswith("f "):
                    # Each token is "v", "v/vt", "v/vt/vn", or "v//vn" —
                    # only the vertex index (first field) is needed since
                    # normals are recomputed from vertex positions anyway.
                    tokens = line.split()[1:]
                    idx = [int(tok.split("/")[0]) - 1 for tok in tokens]  # OBJ is 1-indexed
                    # Fan-triangulate faces with more than 3 vertices.
                    for k in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[k], idx[k + 1]])
        if not verts:
            raise ValueError(f"No vertices found while parsing '{path.name}' as OBJ.")
        verts = np.array(verts, dtype=np.float64)
        faces_arr = np.array(faces, dtype=np.int64) if faces else None
        return verts, faces_arr

    raise ValueError(
        f"geometry_from_file fallback parser doesn't support '{suffix}' files "
        f"without trimesh installed. Run `pip install trimesh` for support of "
        f".ply, .off, and any other mesh format."
    )


# ============================================================================
# 7. Boundary conditions on arbitrary geometry
# ============================================================================

BCKind = str  # "dirichlet" | "neumann" | "robin" | "periodic"


@dataclass
class BoundaryCondition:
    """
    A single boundary condition applied to a subset of ∂Ω.

    Parameters
    ----------
    kind    : "dirichlet" | "neumann" | "robin" | "periodic"
    region  : predicate ``(x) -> bool array`` selecting which boundary
              points this condition applies to (defaults to "everywhere")
    value   : callable ``(x) -> array`` giving the prescribed value
              (Dirichlet), prescribed normal derivative (Neumann), or the
              affine target for Robin conditions. Ignored for periodic.
    coeff_a, coeff_b : Robin condition coefficients: a*u + b*du/dn = value
    pair_region : for "periodic" conditions, the predicate selecting the
                  matching region on the opposite side (e.g. x_min <-> x_max)
    pair_map    : for "periodic" conditions, a callable mapping a point in
                  ``region`` to its corresponding point in ``pair_region``
    name    : optional label, useful for logging / loss breakdown
    """
    kind: BCKind
    value: Optional[Callable[[np.ndarray], np.ndarray]] = None
    region: Callable[[np.ndarray], np.ndarray] = field(default=lambda x: np.ones(len(x), dtype=bool))
    coeff_a: float = 1.0
    coeff_b: float = 0.0
    pair_region: Optional[Callable[[np.ndarray], np.ndarray]] = None
    pair_map: Optional[Callable[[np.ndarray], np.ndarray]] = None
    name: str = ""

    def __post_init__(self) -> None:
        valid = {"dirichlet", "neumann", "robin", "periodic"}
        if self.kind not in valid:
            raise ValueError(f"Unknown BC kind '{self.kind}'. Choose from {sorted(valid)}.")
        if self.kind != "periodic" and self.value is None:
            raise ValueError(f"BoundaryCondition(kind='{self.kind}') requires `value`.")
        if self.kind == "periodic" and (self.pair_region is None or self.pair_map is None):
            raise ValueError("BoundaryCondition(kind='periodic') requires `pair_region` and `pair_map`.")
        if not self.name:
            self.name = f"{self.kind}_bc"


class BoundaryConditionSet:
    """
    A collection of BoundaryConditions covering (possibly disjoint,
    possibly overlapping-and-prioritised) regions of a Geometry's boundary.

    >>> geo = box([(-1, 1), (-1, 1)])
    >>> bcs = BoundaryConditionSet(geo)
    >>> bcs.add("dirichlet", value=lambda x: np.zeros(len(x)),
    ...         region=lambda x: x[:, 0] < -1 + 1e-6, name="left_wall")
    >>> bcs.add("neumann", value=lambda x: np.zeros(len(x)),
    ...         region=lambda x: x[:, 0] > 1 - 1e-6, name="right_wall")
    >>> pts = geo.sample_boundary(1000)
    >>> tagged = bcs.assign(pts)   # dict: bc.name -> points in that region
    """

    def __init__(self, geometry: Geometry) -> None:
        self.geometry = geometry
        self.conditions: List[BoundaryCondition] = []

    def add(self, kind: BCKind, **kwargs: Any) -> BoundaryCondition:
        bc = BoundaryCondition(kind=kind, **kwargs)
        self.conditions.append(bc)
        return bc

    def add_bc(self, bc: BoundaryCondition) -> None:
        self.conditions.append(bc)

    def assign(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Partition boundary points ``x`` by which condition's region they
        fall in. Conditions are checked in the order they were added;
        a point is assigned to the *first* matching condition (so put
        more specific regions — corners, small patches — before broad
        catch-alls like "everywhere").
        """
        x = np.atleast_2d(x)
        remaining = np.ones(len(x), dtype=bool)
        out: Dict[str, np.ndarray] = {}
        for bc in self.conditions:
            mask = remaining & np.asarray(bc.region(x), dtype=bool)
            out[bc.name] = x[mask]
            remaining &= ~mask
        if remaining.any():
            out.setdefault("_unassigned", x[remaining])
        return out

    def residual_targets(self, x: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """
        For each condition, return the points assigned to it together with
        its prescribed target values (and, for Robin, its coefficients) —
        the shape the trainer's loss functions consume directly.
        """
        tagged = self.assign(x)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for bc in self.conditions:
            pts = tagged.get(bc.name, np.empty((0, self.geometry.dim)))
            entry: Dict[str, Any] = {"points": pts, "kind": bc.kind}
            if bc.kind == "periodic":
                entry["pair_points"] = bc.pair_map(pts) if len(pts) else pts
            else:
                entry["target"] = bc.value(pts) if len(pts) else np.empty((0,))
                if bc.kind == "robin":
                    entry["coeff_a"] = bc.coeff_a
                    entry["coeff_b"] = bc.coeff_b
            out[bc.name] = entry
        return out


# Convenience helpers for the most common region predicates ------------------

def face_region(axis: int, side: str, tol: float = 1e-4) -> Callable[[np.ndarray], np.ndarray]:
    """
    Predicate factory for 'the face where coordinate `axis` is at its
    min/max' — the common case for box-like domains.

    >>> region = face_region(axis=0, side="min")   # x == x_min face
    """
    if side not in ("min", "max"):
        raise ValueError("side must be 'min' or 'max'")

    def _region(x: np.ndarray, _axis=axis, _side=side) -> np.ndarray:
        col = x[:, _axis]
        target = col.min() if _side == "min" else col.max()
        return np.abs(col - target) < tol

    return _region


def everywhere(x: np.ndarray) -> np.ndarray:
    """Predicate matching all points — useful as a final catch-all BC."""
    return np.ones(len(x), dtype=bool)


# ============================================================================
# __all__
# ============================================================================

__all__ = [
    "Geometry",
    "box", "ball", "ellipsoid", "cylinder", "torus", "capsule",
    "half_space", "polygon",
    "cone", "frustum", "revolve", "extrude", "tile",
    "rounded_box", "round_geometry", "shell",
    "translate", "scale", "rotate2d", "rotate3d",
    "union", "intersection", "difference", "invert",
    "smooth_union", "smooth_intersection", "smooth_difference",
    "custom_geometry",
    "point_cloud_geometry",
    "geometry_from_file",
    "BoundaryCondition", "BoundaryConditionSet",
    "face_region", "everywhere",
]
