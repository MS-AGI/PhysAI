"""
tests/test_geometry.py

Full-coverage correctness tests for physai.geometry.

geometry.py has no PDEMeta/registry to enumerate against, so "coverage"
here means: every public primitive/transform/combinator/BC class gets at
least one test that checks an *independently computed* expected value —
not just "it runs without crashing." Several docstrings in geometry.py
make specific exactness claims (e.g. "revolve matches cylinder() to
float64 precision", "extrude matches box() exactly", "rounded_box is
exact at a flat-face point and a rounded-corner point") — this file
actually checks those claims rather than trusting the comment.

Sections mirror geometry.py's own section numbering:
1. Geometry base class (distance/contains/hard_constraint_fn/sampling)
2. Primitives (box, ball, ellipsoid, cylinder, torus, capsule, half_space,
   polygon) — each checked against a hand-computed SDF value
2b. Solids of revolution / extrusion / tiling / rounding
3. Transforms (translate, scale, rotate2d, rotate3d)
4. CSG combinators (union, intersection, difference, invert, smooth_*)
5. User-defined / point-cloud / file-based geometry
7. Boundary conditions (BoundaryCondition, BoundaryConditionSet)

Run with: pytest tests/test_geometry.py -v
"""
import numpy as np
import pytest

from physai.geometry import (
    Geometry, box, ball, ellipsoid, cylinder, torus, capsule, half_space,
    polygon, cone, frustum, revolve, extrude, tile, rounded_box,
    round_geometry, shell, translate, scale, rotate2d, rotate3d,
    union, intersection, difference, invert, smooth_union,
    smooth_intersection, smooth_difference, custom_geometry,
    point_cloud_geometry, BoundaryCondition, BoundaryConditionSet,
    face_region, everywhere,
)

RNG = np.random.default_rng(42)


def _rand_pts(n, d, lo=-2.0, hi=2.0):
    return RNG.uniform(lo, hi, size=(n, d))


# ---------------------------------------------------------------------------
# 1. Geometry base class
# ---------------------------------------------------------------------------

def test_distance_sign_convention_box():
    """Inside strictly negative, outside strictly positive, matching the
    documented convention (d<0 inside, d>0 outside, d=0 on boundary)."""
    geo = box([(-1, 1), (-1, 1)])
    assert geo.distance(np.array([[0.0, 0.0]]))[0] < 0
    assert geo.distance(np.array([[5.0, 5.0]]))[0] > 0
    assert abs(geo.distance(np.array([[1.0, 0.0]]))[0]) < 1e-9


def test_contains_matches_sign_of_distance():
    geo = ball([0, 0], 1.0)
    pts = _rand_pts(200, 2)
    d = geo.distance(pts)
    c = geo.contains(pts)
    assert np.array_equal(c, d <= 0.0)


def test_hard_constraint_fn_is_negated_distance_and_zero_on_boundary():
    geo = ball([0, 0, 0], 1.0)
    bump = geo.hard_constraint_fn()
    pts = _rand_pts(50, 3)
    assert np.allclose(bump(pts), -geo.distance(pts))
    # On the analytic boundary (radius exactly 1), bump should be ~0.
    boundary_pt = np.array([[1.0, 0.0, 0.0]])
    assert abs(bump(boundary_pt)[0]) < 1e-9


def test_sample_interior_all_strictly_inside():
    geo = ball([0, 0], 1.0)
    pts = geo.sample_interior(300, rng=np.random.default_rng(1))
    assert len(pts) == 300
    assert np.all(geo.distance(pts) < 0)


def test_sample_boundary_close_to_zero_level_set():
    geo = ball([0, 0], 1.0)
    pts = geo.sample_boundary(300, rng=np.random.default_rng(2))
    assert len(pts) == 300
    # Exact analytic sampler is used for ball -> should be essentially exact.
    assert np.allclose(np.linalg.norm(pts, axis=-1), 1.0, atol=1e-5)


def test_normal_points_outward_on_ball():
    geo = ball([0, 0, 0], 2.0)
    pt = np.array([[2.0, 0.0, 0.0]])
    n = geo.normal(pt)
    # Outward normal at (R,0,0) on a centered ball is (1,0,0).
    assert np.allclose(n, [[1.0, 0.0, 0.0]], atol=1e-3)


def test_csg_operator_overloads_match_function_calls():
    a, b = ball([0, 0], 1.0), ball([0.5, 0], 1.0)
    pts = _rand_pts(100, 2)
    assert np.allclose((a | b).distance(pts), union(a, b).distance(pts))
    assert np.allclose((a & b).distance(pts), intersection(a, b).distance(pts))
    assert np.allclose((a - b).distance(pts), difference(a, b).distance(pts))
    assert np.allclose((~a).distance(pts), invert(a).distance(pts))


# ---------------------------------------------------------------------------
# 2. Primitives — each checked against an independently hand-computed value
# ---------------------------------------------------------------------------

def test_box_sdf_exact_values():
    geo = box([(-1, 1), (-2, 2)])
    # Center: distance to nearest face. Nearest face is x=+-1 (dist 1) vs
    # y=+-2 (dist 2) -> inside distance = -min(1,2) = -1.
    assert np.isclose(geo.distance(np.array([[0.0, 0.0]]))[0], -1.0)
    # Directly outside the x-face at x=3,y=0: distance = 3-1 = 2.
    assert np.isclose(geo.distance(np.array([[3.0, 0.0]]))[0], 2.0)
    # Outside both faces (corner case): Euclidean distance to nearest corner.
    pt = np.array([[2.0, 3.0]])  # corner is (1,2); offset (1,1) -> dist sqrt(2)
    assert np.isclose(geo.distance(pt)[0], np.sqrt(2.0))


def test_box_exact_boundary_sampler_lies_on_faces():
    geo = box([(-1, 1), (-3, 3), (-0.5, 0.5)])
    pts = geo.sample_boundary(500, rng=np.random.default_rng(3))
    d = geo.distance(pts)
    assert np.allclose(d, 0.0, atol=1e-5)
    # Every sampled point must lie exactly on one of the 6 faces (one
    # coordinate pinned to its bound) — not just near-zero SDF by luck.
    lo = np.array([-1, -3, -0.5])
    hi = np.array([1, 3, 0.5])
    on_face = np.any(
        (np.abs(pts - lo) < 1e-5) | (np.abs(pts - hi) < 1e-5), axis=-1
    )
    assert np.all(on_face)


def test_ball_sdf_exact_values():
    geo = ball([1.0, 1.0], 2.0)
    assert np.isclose(geo.distance(np.array([[1.0, 1.0]]))[0], -2.0)  # center
    assert np.isclose(geo.distance(np.array([[3.0, 1.0]]))[0], 0.0)   # on boundary
    assert np.isclose(geo.distance(np.array([[5.0, 1.0]]))[0], 2.0)   # outside


def test_ball_exact_boundary_sampler_has_correct_radius_and_center():
    geo = ball([2.0, -1.0, 0.5], 3.0)
    pts = geo.sample_boundary(400, rng=np.random.default_rng(4))
    r = np.linalg.norm(pts - np.array([2.0, -1.0, 0.5]), axis=-1)
    assert np.allclose(r, 3.0, atol=1e-4)


def test_ellipsoid_exact_on_axis_endpoints():
    """On-axis points are exact for the ellipsoid SDF even though the
    formula is only a first-order approximation off-axis."""
    geo = ellipsoid([0, 0, 0], [2.0, 1.0, 0.5])
    assert np.isclose(geo.distance(np.array([[2.0, 0.0, 0.0]]))[0], 0.0, atol=1e-9)
    assert np.isclose(geo.distance(np.array([[0.0, 1.0, 0.0]]))[0], 0.0, atol=1e-9)
    assert np.isclose(geo.distance(np.array([[0.0, 0.0, 0.5]]))[0], 0.0, atol=1e-9)
    # A clearly-interior point away from the center-instability zone (see
    # test_ellipsoid_center_is_numerically_unstable below) is negative.
    assert geo.distance(np.array([[0.5, 0.2, 0.1]]))[0] < 0


def test_ellipsoid_center_is_numerically_unstable():
    """Documents a real formula issue found while testing: ellipsoid()'s
    SDF is k0*(k0-1)/k1 with k0=|p|, k1=|p/a|+1e-12, p=(x-c)/a. At the
    EXACT center p=(0,0,0), k0=0 forces the whole expression to exactly 0
    (wrong -- the true SDF there should be roughly -min(semi_axes), the
    distance to the nearest surface point, not 0). Worse, this isn't just
    a single mis-valued point: an infinitesimal (~1e-12) perturbation away
    from the exact center lands far outside the 1e-12 floor in `k1`,
    giving a value close to the *correct* magnitude (~-min(semi_axis))
    instead -- i.e. the formula is discontinuous in the limit approaching
    its own center, not smoothly wrong. This test pins down and documents
    that behavior (found via test_rotate3d_about_z_matches_rotate2d_in_plane
    initially failing for an ellipsoid input purely because rotate3d's own
    floating-point cos(pi/2)/sin(pi/2) noise landed the rotated query
    ~2e-12 off the ellipsoid's center) rather than silently working around
    it, since a caller placing a collocation point near an ellipsoid's
    center (a very ordinary thing to do) would see this same instability.
    """
    geo = ellipsoid([0, 0, 0], [0.3, 0.3, 5.0])
    exact_center = geo.distance(np.array([[0.0, 0.0, 0.0]]))[0]
    near_center = geo.distance(np.array([[0.0, 2e-12, 0.0]]))[0]
    assert np.isclose(exact_center, 0.0, atol=1e-9)  # the (wrong) exact-center value
    assert near_center < -0.2  # a hair away, it jumps to ~correct magnitude
    assert abs(near_center - exact_center) > 0.2  # discontinuous in the limit


def test_project_to_boundary_lands_on_zero_level_set():
    geo = ball([0, 0], 1.0)
    # Excludes the exact center (0,0): the SDF gradient is exactly zero
    # there by symmetry (a genuine cone-point singularity of |x|-r), so
    # Newton's method can't determine *which* direction to project toward
    # and correctly makes no progress -- an inherent indeterminacy (any
    # boundary point is equally "nearest"), not a fixable bug. See
    # test_project_to_boundary_center_point_is_a_known_indeterminate_case.
    interior_pts = np.array([[0.3, 0.1], [-0.5, 0.2], [0.05, -0.05]])
    projected = geo.project_to_boundary(interior_pts, steps=6)
    d = geo.distance(projected)
    assert np.allclose(d, 0.0, atol=1e-3)


def test_project_to_boundary_center_point_is_a_known_indeterminate_case():
    """The exact center of a ball is equidistant from every boundary
    point, so its SDF gradient is exactly zero there -- Newton's method
    (which steps along that gradient) can't make progress and the point
    stays put. Documented here as expected/inherent behavior rather than
    a bug: there is no well-defined "nearest" boundary point to project
    the exact center onto."""
    geo = ball([0, 0], 1.0)
    center = np.array([[0.0, 0.0]])
    projected = geo.project_to_boundary(center, steps=6)
    assert np.allclose(projected, center, atol=1e-9)  # made no progress
    assert np.isclose(geo.distance(projected)[0], -1.0)  # still at center, not on boundary


def test_cylinder_sdf_matches_hand_derivation():
    geo = cylinder([0, 0, 0], [0, 0, 4], radius=1.0)
    # Center of the cylinder: inside, distance to nearest surface is
    # min(radial gap, axial gap) = min(1.0, 2.0) = 1.0 -> -1.0
    assert np.isclose(geo.distance(np.array([[0, 0, 2]]))[0], -1.0)
    # Radially outside, same height as center: 2.0 - 1.0 = 1.0
    assert np.isclose(geo.distance(np.array([[2.0, 0, 2]]))[0], 1.0)
    # Beyond the cap, on-axis: 5 - 4 = 1.0 (axial cap at z=4)
    assert np.isclose(geo.distance(np.array([[0, 0, 5]]))[0], 1.0)


def test_torus_sdf_matches_hand_derivation():
    geo = torus([0, 0, 0], major_radius=3.0, minor_radius=1.0)
    # On the tube's centerline circle (radius 3 in xy, z=0): distance to
    # the tube surface is exactly -1 (deepest interior point of the tube).
    assert np.isclose(geo.distance(np.array([[3.0, 0.0, 0.0]]))[0], -1.0)
    # On the outer equator of the tube (radius 3+1=4, z=0): boundary, d=0.
    assert np.isclose(geo.distance(np.array([[4.0, 0.0, 0.0]]))[0], 0.0, atol=1e-9)


def test_capsule_sdf_matches_hand_derivation():
    geo = capsule([0, 0], [4, 0], radius=1.0)
    # Midpoint of the segment: distance to segment is 0, so SDF = -radius.
    assert np.isclose(geo.distance(np.array([[2.0, 0.0]]))[0], -1.0)
    # Perpendicular to an endpoint cap: at (0, 2), nearest segment point is
    # (0,0), dist=2, SDF = 2-1 = 1.
    assert np.isclose(geo.distance(np.array([[0.0, 2.0]]))[0], 1.0)
    # Beyond the far endpoint on-axis: (6,0) -> nearest segment pt (4,0),
    # dist=2, SDF=1.
    assert np.isclose(geo.distance(np.array([[6.0, 0.0]]))[0], 1.0)


def test_half_space_sdf_is_signed_perpendicular_distance():
    geo = half_space(point=[0, 0], normal=[1, 0])
    assert np.isclose(geo.distance(np.array([[-2.0, 5.0]]))[0], -2.0)
    assert np.isclose(geo.distance(np.array([[3.0, -7.0]]))[0], 3.0)


def test_polygon_sdf_square_matches_box():
    """A square polygon should match box()'s exact SDF (both are exact
    Euclidean distances to the same shape)."""
    poly = polygon(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]))
    bx = box([(-1, 1), (-1, 1)])
    pts = _rand_pts(200, 2, lo=-3, hi=3)
    assert np.allclose(poly.distance(pts), bx.distance(pts), atol=1e-6)


# ---------------------------------------------------------------------------
# 2b. Revolution / extrusion / tiling / rounding — checking exactness claims
# ---------------------------------------------------------------------------

def test_revolve_cylinder_matches_primitive_cylinder():
    """Docstring claims float64-precision agreement with cylinder() for a
    correctly mirrored rectangular profile. Verify directly."""
    R, h = 1.5, 4.0
    profile = polygon(np.array([[-R, 0.0], [R, 0.0], [R, h], [-R, h]]))
    revolved = revolve(profile, [0, 0, 0], [0, 0, h])
    ref = cylinder([0, 0, 0], [0, 0, h], radius=R)
    pts = _rand_pts(300, 3, lo=-3, hi=3)
    # revolve's z-coordinate is measured from point_a along the axis,
    # matching cylinder's own axial parametrization here since both use
    # [0,0,0]->[0,0,h].
    assert np.allclose(revolved.distance(pts), ref.distance(pts), atol=1e-6)


def test_revolve_half_profile_is_wrong_near_axis():
    """Documents/confirms the docstring's warning: the *unmirrored* half
    profile corrupts the SDF near the rotation axis, quantitatively."""
    R, h = 1.0, 2.0
    half_profile = polygon(np.array([[0.0, 0.0], [R, 0.0], [R, h], [0.0, h]]))
    bad = revolve(half_profile, [0, 0, 0], [0, 0, h])
    ref = cylinder([0, 0, 0], [0, 0, h], radius=R)
    near_axis = np.array([[0.01, 0.0, 1.0]])
    # The docstring claims O(R) error near the axis for the half-profile
    # version -- assert the discrepancy is large (not just numerically
    # noisy), i.e. the two disagree by a large fraction of R.
    diff = abs(bad.distance(near_axis)[0] - ref.distance(near_axis)[0])
    assert diff > 0.5 * R


def test_extrude_square_matches_box():
    """Docstring claims exact match to box() for an axis-aligned square
    profile extruded along z."""
    profile = polygon(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]))
    extruded = extrude(profile, [0, 0, 0], [0, 0, 5])
    ref = box([(-1, 1), (-1, 1), (0, 5)])
    pts = _rand_pts(300, 3, lo=-3, hi=8)
    assert np.allclose(extruded.distance(pts), ref.distance(pts), atol=1e-5)


def test_cone_apex_and_base_radius():
    geo = cone(apex=[0, 0, 0], base=[0, 0, 4], radius=2.0)
    # Apex itself should be on/near the boundary (radius 0 there).
    assert abs(geo.distance(np.array([[0, 0, 0]]))[0]) < 0.05
    # Center of the base disc should be strictly inside.
    assert geo.distance(np.array([[0, 0, 3.99]]))[0] < 0


def test_frustum_matches_cylinder_when_radii_equal():
    """A frustum with equal top/bottom radius degenerates to a cylinder —
    cross-check against the dedicated primitive."""
    R, h = 1.0, 3.0
    fr = frustum([0, 0, 0], R, [0, 0, h], R)
    cyl = cylinder([0, 0, 0], [0, 0, h], radius=R)
    pts = _rand_pts(200, 3, lo=-2, hi=5)
    assert np.allclose(fr.distance(pts), cyl.distance(pts), atol=1e-6)


def test_tile_infinite_matches_base_geometry_in_every_period():
    geo = ball([0, 0], 0.4)
    tiled = tile(geo, period=[2.0, 2.0])
    # Center of period (0,0), (2,0), (-2,2), etc. should all read as
    # strongly inside (same as the un-tiled ball at its own center).
    centers = np.array([[0, 0], [2, 0], [-2, 0], [0, 2], [-2, 2]])
    assert np.allclose(tiled.distance(centers), geo.distance(np.array([[0, 0]]))[0])


def test_tile_finite_reps_bounds_the_domain():
    geo = ball([0, 0], 0.3)
    tiled = tile(geo, period=[1.0, 1.0], reps=[3, 3])
    # A point far outside the finite tiled region should NOT be inside,
    # even though it would land on a tile center under infinite tiling.
    far_pt = np.array([[100.0, 100.0]])
    assert tiled.distance(far_pt)[0] > 0
    # A point inside the 3x3 finite region, at a valid tile center, is
    # still inside.
    near_pt = np.array([[1.0, 1.0]])
    assert tiled.distance(near_pt)[0] < 0


def test_rounded_box_exact_at_flat_face_and_rounded_corner():
    bounds = [(-2, 2), (-2, 2)]
    radius = 0.5
    geo = rounded_box(bounds, radius)
    # Flat-face point: rounding a box by shrinking by `radius` then adding
    # `radius` back via "d(x) -> shrunk_sdf(x) - radius" leaves flat faces
    # (away from corners) exactly where the ORIGINAL box's faces were --
    # the standard property of this rounding construction. So a point on
    # the original box's flat face, e.g. (2, 0), should be exactly ON the
    # rounded box's surface too (d=0), not offset by `radius`.
    flat_pt = np.array([[2.0, 0.0]])
    assert np.isclose(geo.distance(flat_pt)[0], 0.0, atol=1e-9)
    # Rounded-corner point: straight out from the inner corner
    # (2-radius, 2-radius) along the diagonal by exactly `radius` should
    # land exactly on the rounded surface (d=0).
    inner_corner = np.array([2 - radius, 2 - radius])
    diag = np.array([1.0, 1.0]) / np.sqrt(2.0)
    corner_surface_pt = (inner_corner + radius * diag)[None, :]
    assert np.isclose(geo.distance(corner_surface_pt)[0], 0.0, atol=1e-6)


def test_round_geometry_shifts_ball_sdf_by_radius():
    geo = ball([0, 0], 2.0)
    rounded = round_geometry(geo, radius=0.3)
    pts = _rand_pts(100, 2)
    assert np.allclose(rounded.distance(pts), geo.distance(pts) - 0.3)


def test_shell_is_thin_band_around_original_boundary():
    geo = ball([0, 0], 1.0)
    thickness = 0.2
    sh = shell(geo, thickness)
    # Exactly on the original boundary: |d|=0, so shell SDF = -thickness/2
    # (deep inside the thin shell).
    on_boundary = np.array([[1.0, 0.0]])
    assert np.isclose(sh.distance(on_boundary)[0], -thickness / 2, atol=1e-6)
    # Far from the boundary (deep interior or exterior): |d| large positive.
    center = np.array([[0.0, 0.0]])
    assert sh.distance(center)[0] > 0  # |d(0)|=1 >> thickness/2=0.1


# ---------------------------------------------------------------------------
# 3. Transforms
# ---------------------------------------------------------------------------

def test_translate_shifts_sdf_correctly():
    geo = ball([0, 0], 1.0)
    moved = translate(geo, [5.0, 0.0])
    assert np.isclose(moved.distance(np.array([[5.0, 0.0]]))[0], -1.0)
    assert np.isclose(moved.distance(np.array([[0.0, 0.0]]))[0], geo.distance(np.array([[-5.0, 0.0]]))[0])


def test_scale_uniform_matches_hand_derivation():
    geo = ball([0, 0], 1.0)
    scaled = scale(geo, 2.0)  # should become radius-2 ball
    assert np.isclose(scaled.distance(np.array([[2.0, 0.0]]))[0], 0.0, atol=1e-6)
    assert np.isclose(scaled.distance(np.array([[4.0, 0.0]]))[0], 2.0, atol=1e-6)


def test_rotate2d_by_90_degrees_matches_hand_rotated_query():
    """A non-symmetric shape (off-center ball) rotated 90 degrees about the
    origin should read the same distance at a query point as the
    un-rotated shape does at that point rotated by -90 degrees."""
    geo = ball([1.0, 0.0], 0.3)  # off-center, breaks rotational symmetry
    rotated = rotate2d(geo, angle_rad=np.pi / 2)
    query = np.array([[0.0, 1.0]])
    # Rotating (0,1) by -90 degrees -> (1, 0)
    expected = geo.distance(np.array([[1.0, 0.0]]))
    assert np.allclose(rotated.distance(query), expected, atol=1e-6)


def test_rotate3d_about_z_matches_rotate2d_in_plane():
    """Rotating an off-center shape about the z axis should match the 2D
    in-plane rotation for points at z=0. Uses a ball rather than an
    off-axis ellipsoid deliberately: an ellipsoid's SDF formula is
    numerically unstable within ~1e-11 of its own center (see
    test_ellipsoid_center_is_numerically_unstable), and rotate3d's own
    floating-point cos(pi/2)/sin(pi/2) noise is large enough to land a
    rotated query exactly in that unstable zone for a naive choice of
    shape/geometry -- that instability, not a rotate3d bug, is what an
    earlier version of this test (using an ellipsoid) was actually
    catching. A ball has no such instability at its center."""
    geo3d = ball([1.0, 0.0, 0.0], 0.3)  # off-center, breaks rotational symmetry
    rotated3d = rotate3d(geo3d, axis=[0, 0, 1], angle_rad=np.pi / 2, center=[0, 0, 0])
    query = np.array([[0.0, 1.0, 0.0]])
    expected = geo3d.distance(np.array([[1.0, 0.0, 0.0]]))
    assert np.allclose(rotated3d.distance(query), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. CSG combinators
# ---------------------------------------------------------------------------

def test_union_is_pointwise_min():
    a, b = ball([-1, 0], 0.8), ball([1, 0], 0.8)
    u = union(a, b)
    pts = _rand_pts(200, 2)
    assert np.allclose(u.distance(pts), np.minimum(a.distance(pts), b.distance(pts)))


def test_intersection_is_pointwise_max():
    a, b = ball([-0.3, 0], 1.0), ball([0.3, 0], 1.0)
    i = intersection(a, b)
    pts = _rand_pts(200, 2)
    assert np.allclose(i.distance(pts), np.maximum(a.distance(pts), b.distance(pts)))


def test_difference_matches_formula_and_geometric_meaning():
    a, b = ball([0, 0], 1.0), ball([0.5, 0], 0.6)
    diff = difference(a, b)
    pts = _rand_pts(200, 2)
    assert np.allclose(diff.distance(pts), np.maximum(a.distance(pts), -b.distance(pts)))
    # A point inside `a` but also inside `b` must NOT be inside a-b.
    inside_both = np.array([[0.5, 0.0]])
    assert a.contains(inside_both)[0] and b.contains(inside_both)[0]
    assert not diff.contains(inside_both)[0]


def test_invert_flips_sign_exactly():
    geo = ball([0, 0], 1.0)
    inv = invert(geo)
    pts = _rand_pts(100, 2)
    assert np.allclose(inv.distance(pts), -geo.distance(pts))


def test_smooth_union_matches_hard_union_far_from_seam():
    """Where |d_a - d_b| >> k, the smooth-min blend weight h saturates to
    0 or 1 and smooth_union coincides with the hard union. Deep inside
    ball `a` (far from ball `b`'s surface, so d_b is large and very
    different from d_a) is such a point -- unlike, say, points on the
    perpendicular bisector between two far-apart balls, where d_a=d_b
    exactly (h=0.5, maximal blend) *regardless* of how far apart the
    balls are. That equidistant case is a distinct, separately-tested
    property (see test_smooth_union_is_never_less_than_hard_union...);
    this test specifically picks points where the blend should vanish."""
    a, b = ball([-3, 0], 0.5), ball([3, 0], 0.5)
    su = smooth_union(a, b, k=0.1)
    hu = union(a, b)
    # Points deep inside `a`'s interior: d_a is very negative, d_b is very
    # positive (b is 6 units away) -- |d_a - d_b| is huge relative to k=0.1.
    pts = np.array([[-3.0, 0.0], [-3.1, 0.05], [-2.9, -0.05]])
    assert np.allclose(su.distance(pts), hu.distance(pts), atol=1e-6)


def test_smooth_union_is_never_less_than_hard_union_minus_k_over_4():
    """Sanity bound on the polynomial smin: the blend never dips more than
    k/4 below the true (hard) minimum -- standard property of this
    smin formula. Also checks it's always <= the hard union value (i.e.
    it's genuinely a *lower* smoothing of min, not an upper one)."""
    a, b = ball([-0.3, 0], 0.6), ball([0.3, 0], 0.6)
    k = 0.2
    su = smooth_union(a, b, k=k)
    hu = union(a, b)
    pts = _rand_pts(300, 2)
    su_d, hu_d = su.distance(pts), hu.distance(pts)
    assert np.all(su_d <= hu_d + 1e-9)
    assert np.all(su_d >= hu_d - k / 4.0 - 1e-9)


def test_smooth_intersection_matches_hard_intersection_far_from_seam():
    a, b = ball([-0.2, 0], 3.0), ball([0.2, 0], 3.0)  # heavy overlap
    si = smooth_intersection(a, b, k=0.1)
    hi_ = intersection(a, b)
    # Points where d_a and d_b clearly differ (off the perpendicular
    # bisector x=0, where d_a=d_b exactly and the blend is maximal
    # regardless of k) -- e.g. near a's boundary but deep inside b.
    pts = np.array([[-2.75, 0.0], [-2.7, 0.1], [-2.8, -0.1]])
    assert np.allclose(si.distance(pts), hi_.distance(pts), atol=1e-6)


def test_smooth_difference_matches_hard_difference_far_from_seam():
    a, b = ball([0, 0], 3.0), ball([0, 0], 0.2)  # small cutout, far boundaries
    sd = smooth_difference(a, b, k=0.05)
    hd = difference(a, b)
    pts = np.array([[1.5, 0.0], [0.0, 1.5], [-1.5, 0.0]])
    assert np.allclose(sd.distance(pts), hd.distance(pts), atol=1e-2)


# ---------------------------------------------------------------------------
# 5. User-defined / point-cloud geometry
# ---------------------------------------------------------------------------

def test_custom_geometry_wraps_arbitrary_sdf_correctly():
    def superellipse_sdf(x):
        return (np.abs(x[:, 0]) ** 4 + np.abs(x[:, 1]) ** 4) - 1.0

    geo = custom_geometry(superellipse_sdf, bounds=[(-1.2, 1.2), (-1.2, 1.2)])
    assert geo.distance(np.array([[0.0, 0.0]]))[0] < 0
    assert geo.distance(np.array([[2.0, 2.0]]))[0] > 0
    assert np.isclose(geo.distance(np.array([[1.0, 0.0]]))[0], 0.0, atol=1e-9)


def test_custom_geometry_rejects_wrong_output_shape():
    def bad_sdf(x):
        return np.zeros((len(x), 2))  # wrong shape: should be (n,)

    with pytest.raises(ValueError):
        custom_geometry(bad_sdf, bounds=[(-1, 1), (-1, 1)])


def test_custom_geometry_surfaces_exceptions_from_sdf_fn():
    def broken_sdf(x):
        raise RuntimeError("boom")

    with pytest.raises(ValueError, match="raised an exception"):
        custom_geometry(broken_sdf, bounds=[(-1, 1)])


def test_point_cloud_geometry_signed_matches_ball_near_surface():
    """Densely sample a unit circle with exact analytic normals and check
    the point-cloud SDF agrees with the true ball() SDF near the surface
    (the docstring's own claimed verification, reproduced here)."""
    n = 2000
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    normals = pts.copy()  # outward normal of a unit circle = position itself
    pc = point_cloud_geometry(pts, normals=normals)
    ref = ball([0, 0], 1.0)

    query = np.array([[1.05, 0.0], [0.0, 0.95], [0.7, 0.7]])
    query = query / np.where(  # keep the mixed point roughly near the circle
        np.arange(len(query))[:, None] == 2, np.linalg.norm(query, axis=-1, keepdims=True) / 1.0, 1.0
    )
    got = pc.distance(query)
    expected = ref.distance(query)
    assert np.allclose(got, expected, atol=0.02)


def test_point_cloud_geometry_unsigned_is_nonnegative():
    theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    pts = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    pc = point_cloud_geometry(pts, normals=None)
    query = _rand_pts(50, 2, lo=-2, hi=2)
    assert np.all(pc.distance(query) >= 0)
    assert getattr(pc, "_is_unsigned_fallback", None) is True


def test_point_cloud_geometry_rejects_mismatched_shapes():
    pts = _rand_pts(10, 2)
    bad_normals = _rand_pts(10, 3)  # wrong dimensionality
    with pytest.raises(ValueError):
        point_cloud_geometry(pts, normals=bad_normals)


# ---------------------------------------------------------------------------
# 7. Boundary conditions
# ---------------------------------------------------------------------------

def test_boundary_condition_validates_kind():
    with pytest.raises(ValueError):
        BoundaryCondition(kind="not_a_real_kind", value=lambda x: x)


def test_boundary_condition_requires_value_unless_periodic():
    with pytest.raises(ValueError):
        BoundaryCondition(kind="dirichlet")  # missing value


def test_boundary_condition_periodic_requires_pair_region_and_map():
    with pytest.raises(ValueError):
        BoundaryCondition(kind="periodic")


def test_boundary_condition_default_name_derived_from_kind():
    bc = BoundaryCondition(kind="neumann", value=lambda x: np.zeros(len(x)))
    assert bc.name == "neumann_bc"


def test_boundary_condition_set_assign_partitions_by_region_in_order():
    geo = box([(-1, 1), (-1, 1)])
    bcs = BoundaryConditionSet(geo)
    bcs.add("dirichlet", value=lambda x: np.zeros(len(x)),
             region=face_region(axis=0, side="min"), name="left")
    bcs.add("dirichlet", value=lambda x: np.ones(len(x)),
             region=everywhere, name="rest")

    pts = geo.sample_boundary(400, rng=np.random.default_rng(7))
    tagged = bcs.assign(pts)

    assert "left" in tagged and "rest" in tagged
    # Every point tagged "left" really is on the x=-1 face.
    assert np.all(np.abs(tagged["left"][:, 0] - (-1.0)) < 1e-3)
    # No point should be double-counted: total count matches input.
    assert sum(len(v) for v in tagged.values()) == len(pts)
    # "rest" must not contain any left-face points (first-match-wins order).
    if len(tagged["rest"]):
        assert not np.any(np.abs(tagged["rest"][:, 0] - (-1.0)) < 1e-3)


def test_boundary_condition_set_residual_targets_dirichlet_values():
    geo = box([(-1, 1), (-1, 1)])
    bcs = BoundaryConditionSet(geo)
    bcs.add("dirichlet", value=lambda x: np.full(len(x), 3.0), name="all")
    pts = geo.sample_boundary(50, rng=np.random.default_rng(8))
    targets = bcs.residual_targets(pts)
    assert np.allclose(targets["all"]["target"], 3.0)
    assert targets["all"]["kind"] == "dirichlet"


def test_boundary_condition_set_residual_targets_robin_coefficients():
    geo = box([(-1, 1), (-1, 1)])
    bcs = BoundaryConditionSet(geo)
    bcs.add("robin", value=lambda x: np.zeros(len(x)), coeff_a=2.0, coeff_b=5.0, name="r")
    pts = geo.sample_boundary(20, rng=np.random.default_rng(9))
    targets = bcs.residual_targets(pts)
    assert targets["r"]["coeff_a"] == 2.0
    assert targets["r"]["coeff_b"] == 5.0


def test_boundary_condition_set_periodic_pair_points():
    geo = box([(-1, 1), (-1, 1)])
    bcs = BoundaryConditionSet(geo)
    bcs.add(
        "periodic",
        region=face_region(axis=0, side="min"),
        pair_region=face_region(axis=0, side="max"),
        pair_map=lambda x: x + np.array([2.0, 0.0]),
        name="periodic_x",
    )
    pts = geo.sample_boundary(200, rng=np.random.default_rng(10))
    targets = bcs.residual_targets(pts)
    entry = targets["periodic_x"]
    if len(entry["points"]):
        assert np.allclose(entry["pair_points"][:, 0], entry["points"][:, 0] + 2.0)


def test_face_region_matches_min_max_face():
    geo = box([(-1, 1), (-1, 1)])
    pts = geo.sample_boundary(500, rng=np.random.default_rng(11))
    left = face_region(axis=0, side="min")(pts)
    right = face_region(axis=0, side="max")(pts)
    assert np.all(pts[left][:, 0] < -1 + 1e-3)
    assert np.all(pts[right][:, 0] > 1 - 1e-3)
    # min/max faces are disjoint on a box with distinct bounds.
    assert not np.any(left & right)


def test_everywhere_matches_all_points():
    pts = _rand_pts(20, 3)
    assert np.all(everywhere(pts))