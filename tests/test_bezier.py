from __future__ import annotations

import numpy as np
import pytest

from glakit.bezier import BezierCubic, BSpline
from glakit.geometry import Point, PointArray
from glakit.transform import Translation


class TestBezierCubic:
    def test_creation_and_properties(self):
        c = BezierCubic((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0))
        assert c.shape == (4, 2)
        assert isinstance(c.p0, Point) and c.p0 == Point(0.0, 0.0)
        assert isinstance(c.c0, Point) and c.c0 == Point(0.0, 10.0)
        assert isinstance(c.c1, Point) and c.c1 == Point(10.0, 10.0)
        assert isinstance(c.p1, Point) and c.p1 == Point(10.0, 0.0)
        assert c.tuple == ((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0))
        assert c.ndarray.shape == (4, 2)

    def test_as_bcubic_conversions_and_errors(self):
        arr = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        c1 = BezierCubic.as_bcubic(arr)
        assert isinstance(c1, BezierCubic)
        assert np.allclose(c1, arr)

        assert BezierCubic.as_bcubic(c1, copy=False) is c1
        c2 = BezierCubic.as_bcubic(c1, copy=True)
        assert c2 is not c1

        with pytest.raises((AssertionError, ValueError)):
            BezierCubic.as_bcubic(np.zeros((3, 2)))

        with pytest.raises((AssertionError, ValueError)):
            BezierCubic.as_bcubic(((0, 0), (1, 1)))  # type: ignore

        with pytest.raises(TypeError):
            BezierCubic.as_bcubic("invalid")  # type: ignore

    def test_setters_and_indexing_slicing(self):
        c = BezierCubic((0, 0), (1, 1), (2, 2), (3, 3))
        c.p0 = Point(10, 20)
        c.c0 = Point(30, 40)
        c.c1 = Point(50, 60)
        c.p1 = Point(70, 80)

        assert c.p0 == Point(10, 20)
        assert c.c0 == Point(30, 40)
        assert c.c1 == Point(50, 60)
        assert c.p1 == Point(70, 80)

        assert isinstance(c[0], Point)
        sub = c[:2]
        assert isinstance(sub, PointArray)
        assert not isinstance(sub, BezierCubic)
        assert sub.shape == (2, 2)

    def test_arithmetic_ufuncs_preserve_bezier_type(self):
        c = BezierCubic((0, 0), (0, 10), (10, 10), (10, 0))
        shifted = c + Point(5.0, -5.0)
        assert isinstance(shifted, BezierCubic)
        assert shifted.p0 == Point(5.0, -5.0)

        scaled = c * 2.0
        assert isinstance(scaled, BezierCubic)
        assert scaled.c1 == Point(20.0, 20.0)

    def test_evaluation_and_tangents(self):
        line = BezierCubic((0, 0), (0, 3), (0, 7), (0, 10))
        pts = line.evaluate(np.array([0.0, 0.5, 1.0]))
        assert isinstance(pts, PointArray)
        assert pts.shape == (3, 2)
        assert pts[0] == Point(0.0, 0.0)
        assert pts[2] == Point(0.0, 10.0)

        uniform_line = BezierCubic((0, 0), (0, 10 / 3), (0, 20 / 3), (0, 10))
        mid = uniform_line.evaluate(np.array([0.5]))
        assert mid[0] == Point(0.0, 5.0)

        tangents = uniform_line.evaluate_tangent(np.array([0.0, 0.5, 1.0]), normalized=False)
        assert isinstance(tangents, PointArray)
        np.testing.assert_allclose(tangents, [[0.0, 10.0]] * 3)

        norm_tangents = uniform_line.evaluate_tangent(np.array([0.0, 0.5, 1.0]), normalized=True)
        np.testing.assert_allclose(norm_tangents, [[0.0, 1.0]] * 3)

    def test_lengths_split_flip_and_symmetry(self):
        line = BezierCubic((0, 0), (0, 3), (0, 7), (0, 10))
        assert line.chord_length() == pytest.approx(10.0)
        assert line.arc_length(fast_approximation=True) == pytest.approx(10.0)
        assert line.arc_length(fast_approximation=False) == pytest.approx(10.0, rel=1e-2)

        flipped = line.flip()
        assert flipped.p0 == line.p1
        assert flipped.c0 == line.c1
        assert flipped.c1 == line.c0
        assert flipped.p1 == line.p0

        left, right = line.split(0.5)
        assert isinstance(left, BezierCubic) and isinstance(right, BezierCubic)
        assert left.p0 == line.p0
        assert left.p1 == right.p0
        assert right.p1 == line.p1

        c = BezierCubic((0, 0), (0, 4), (10, 6), (10, 10))
        assert c.c0_sym(d=-1.0, relative=True) == Point(0.0, -4.0)
        assert c.c0_sym(d=2.0, relative=False) == Point(0.0, 2.0)
        assert c.c1_sym(d=-1.0, relative=True) == Point(10.0, 14.0)
        assert c.c1_sym(d=3.0, relative=False) == Point(10.0, 7.0)

    def test_null_nan_and_svg_path(self):
        null_c = BezierCubic((5, 5), (5, 5), (5, 5), (5, 5))
        assert null_c.is_null()
        assert not null_c.has_nan()

        nan_c = BezierCubic((0, 0), (float("nan"), 1), (2, 2), (3, 3))
        assert nan_c.has_nan()
        assert nan_c.to_path() == ""

        c = BezierCubic((10, 20), (30, 40), (50, 60), (70, 80))
        assert c.to_path() == "M 20.0,10.0 C 40.0,30.0 60.0,50.0 80.0,70.0"
        assert c.to_path(offset=Point(5, 10)) == "M 10.0,5.0 C 30.0,25.0 50.0,45.0 70.0,65.0"

    def test_union_operator_creates_bspline(self):
        c1 = BezierCubic((0, 0), (0, 1), (0, 2), (0, 3))
        c2 = BezierCubic((0, 3), (0, 4), (0, 5), (0, 6))
        spline = c1 | c2
        assert isinstance(spline, BSpline)
        assert len(spline) == 2
        assert np.all(spline[0] == c1)
        assert np.all(spline[1] == c2)

    def test_projection(self):
        # Segment horizontal (y=0, x de 0 à 10)
        line = BezierCubic((0, 0), (0, 10 / 3), (0, 20 / 3), (0, 10))
        query = PointArray([(4.0, 2.5), (-3.0, 7.5), (1.0, -2.0), (1.0, 12.0)])

        proj_fast, u_fast = line.projection(query, fast_approximation=True, return_u=True)
        assert isinstance(proj_fast, PointArray)
        assert proj_fast.shape == (4, 2)
        np.testing.assert_allclose(proj_fast, [[0.0, 2.5], [0.0, 7.5], [0.0, 0.0], [0.0, 10.0]])
        np.testing.assert_allclose(u_fast, [0.25, 0.75, 0.0, 1.0])

        # Mode exact (Newton-Raphson) sans return_u
        proj_exact = line.projection(query, fast_approximation=False, return_u=False)
        assert isinstance(proj_exact, PointArray)
        np.testing.assert_allclose(proj_exact, [[0.0, 2.5], [0.0, 7.5], [0.0, 0.0], [0.0, 10.0]], atol=1e-4)

        # Courbe dégénérée (tous les points de contrôle confondus)
        null_c = BezierCubic((3, 4), (3, 4), (3, 4), (3, 4))
        proj_null, u_null = null_c.projection(PointArray([(0, 0), (10, 10)]), return_u=True)
        np.testing.assert_allclose(proj_null, [[3.0, 4.0], [3.0, 4.0]])
        np.testing.assert_allclose(u_null, [0.0, 0.0])

        # Cas d'une BSpline vide
        empty = BSpline([])
        proj_empty, u_empty = empty.projection(PointArray([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]), return_u=True)
        assert isinstance(proj_empty, PointArray)
        assert proj_empty.shape == (3, 2)
        assert np.all(np.isnan(proj_empty))
        assert u_empty.shape == (3,)
        assert np.all(np.isnan(u_empty))


class TestBSpline:
    @pytest.fixture
    def sample_spline(self) -> BSpline:
        c1 = BezierCubic((0, 0), (0, 10 / 3), (0, 20 / 3), (0, 10))
        c2 = BezierCubic((0, 10), (10 / 3, 10), (20 / 3, 10), (10, 10))
        return BSpline([c1, c2])

    def test_creation_indexing_and_concatenation(self, sample_spline: BSpline):
        assert len(sample_spline) == 2
        assert sample_spline.shape == (2, 4, 2)

        assert isinstance(sample_spline[0], BezierCubic)
        assert isinstance(sample_spline[0, 0], Point)
        assert isinstance(sample_spline[:1], BSpline)
        assert len(sample_spline[:1]) == 1

        curves = list(sample_spline)
        assert len(curves) == 2
        np.testing.assert_array_equal(BSpline(curves), sample_spline)

        empty = BSpline([])
        assert len(empty) == 0
        assert empty.shape == (0, 4, 2)
        assert empty.chord_length() == 0

        c3 = BezierCubic((10, 10), (10, 5), (10, 2), (10, 0))
        extended = [c3] | sample_spline | [c3]
        assert isinstance(extended, BSpline)
        assert len(extended) == 4

    def test_evaluate_and_relative_pos(self, sample_spline: BSpline):
        p_mid_c1 = sample_spline.evaluate(0.5)
        assert isinstance(p_mid_c1, Point)
        assert p_mid_c1 == Point(0.0, 5.0)

        p_mid_c2 = sample_spline.evaluate(1.5)
        assert p_mid_c2 == Point(5.0, 10.0)

        pts = sample_spline.evaluate(np.array([0.0, 1.0, 2.0]))
        assert isinstance(pts, PointArray)
        assert pts[0] == Point(0.0, 0.0)
        assert pts[1] == Point(0.0, 10.0)
        assert pts[2] == Point(10.0, 10.0)

        tan = sample_spline.evaluate_tangent(0.5, normalized=True)
        assert isinstance(tan, Point)
        assert tan == Point(0.0, 1.0)

        t0, t1 = sample_spline.tips_tangents(normalize=True)
        assert t0 == Point(0.0, 1.0)
        assert t1 == Point(-1.0, 0.0)

        assert sample_spline.arc_length() == pytest.approx(20.0, rel=1e-2)
        assert sample_spline.chord_length() == pytest.approx(np.hypot(10.0, 10.0))

        t_quarter = sample_spline.relative_pos_to_t(0.25)
        assert t_quarter == pytest.approx(0.5, abs=1e-2)

        ts = sample_spline.relative_pos_to_t(np.array([0.0, 0.5, 0.75, 1.0]))
        np.testing.assert_allclose(ts, [0.0, 1.0, 1.5, 2.0], atol=1e-2)

    def test_flip_filling_and_interpolate(self):
        c1 = BezierCubic((0, 0), (0, 2), (0, 4), (0, 6))
        c2 = BezierCubic((0, 10), (0, 12), (0, 14), (0, 16))
        disjoint = BSpline([c1, c2])

        flipped = disjoint.flip()
        np.testing.assert_array_equal(flipped[0], c2.flip())
        np.testing.assert_array_equal(flipped[1], c1.flip())

        extrapolated = disjoint.interpolate_bspline(start=Point(0, -4), end=Point(0, 20), smoothing=0.3)
        assert len(extrapolated) == 5
        assert extrapolated[0].p0 == extrapolated[0].c0 == Point(0, -4)
        assert extrapolated[0].p1 == extrapolated[1].p0 == c1.p0
        assert extrapolated[1].p1 == extrapolated[2].p0 == c1.p1
        assert extrapolated[2].p1 == extrapolated[3].p0 == c2.p0
        assert extrapolated[3].p1 == extrapolated[4].p0 == c2.p1
        assert extrapolated[4].p1 == extrapolated[4].c1 == Point(0, 20)

        interpolated = disjoint.interpolate_bspline(smoothing=0.3)
        assert len(interpolated) == 3
        assert interpolated[0].p0 == c1.p0
        assert interpolated[0].p1 == interpolated[1].p0 == c1.p1
        assert interpolated[1].p1 == interpolated[2].p0 == c2.p0
        assert interpolated[2].p1 == c2.p1

        interpolated = disjoint.interpolate_bspline(start=Point(0, -4), smoothing=0.3)
        assert len(interpolated) == 4

    def test_transform_and_svg_path(self, sample_spline: BSpline):
        trans = Translation((5.5, -2.5))
        moved = sample_spline.transform(trans, round_p=False)
        assert isinstance(moved, BSpline)
        assert moved[0].p0 == Point(5.5, -2.5)
        assert moved[1].p1 == Point(15.5, 7.5)

        moved_rounded = sample_spline.transform(trans, round_p=True)
        assert moved_rounded[0].p0 == Point(6.0, -2.0)
        assert moved_rounded[1].p1 == Point(16.0, 8.0)

        svg = sample_spline.to_path()
        np.testing.assert_array_equal(BSpline(svg), sample_spline)

    def test_projection(self, sample_spline: BSpline):
        # sample_spline : c1 de (0, 0) -> (0, 10), c2 de (0, 10) -> (10, 10)
        queries = PointArray(
            [
                (2.0, 3.0),  # proche de c1 à u = 0.3
                (4.0, 12.0),  # proche de c2 à u = 1.4
                (-1.0, -2.0),  # avant le début -> u = 0.0
                (12.0, 11.0),  # après la fin -> u = 2.0
            ]
        )

        proj, u = sample_spline.projection(queries, return_u=True)
        assert isinstance(proj, PointArray)
        assert proj.shape == (4, 2)
        np.testing.assert_allclose(proj, [[0.0, 3.0], [4.0, 10.0], [0.0, 0.0], [10.0, 10.0]], atol=1e-3)
        np.testing.assert_allclose(u, [0.3, 1.4, 0.0, 2.0], atol=1e-3)

        # Vérifie return_u=False et fast_approximation=True
        proj_only = sample_spline.projection(queries, fast_approximation=True, return_u=False)
        assert isinstance(proj_only, PointArray)
        np.testing.assert_allclose(proj_only, [[0.0, 3.0], [4.0, 10.0], [0.0, 0.0], [10.0, 10.0]], atol=1e-3)

        # Cas d'une BSpline à une seule courbe
        single = BSpline([sample_spline[0]])
        proj_single, u_single = single.projection(PointArray([(2.0, 6.0)]), return_u=True)
        np.testing.assert_allclose(proj_single, [[0.0, 6.0]], atol=1e-3)
        np.testing.assert_allclose(u_single, [0.6], atol=1e-3)

    def test_split_and_split_into_multiple_bsplines(self, sample_spline: BSpline):
        # 1. Découpe en plusieurs BSplines par paramètre u (milieu de c1 et milieu de c2)
        parts = sample_spline.split_into_multiple_bsplines(u=np.array([0.5, 1.5]))
        assert len(parts) == 3
        assert all(isinstance(p, BSpline) for p in parts)

        assert len(parts[0]) == 1 and parts[0][0].p0 == Point(0.0, 0.0) and parts[0][0].p1 == Point(0.0, 5.0)
        assert len(parts[1]) == 2
        assert parts[1][0].p0 == Point(0.0, 5.0) and parts[1][0].p1 == Point(0.0, 10.0)
        assert parts[1][1].p0 == Point(0.0, 10.0) and parts[1][1].p1 == Point(5.0, 10.0)
        assert len(parts[2]) == 1 and parts[2][0].p0 == Point(5.0, 10.0) and parts[2][0].p1 == Point(10.0, 10.0)

        # 2. Découpe par coordonnées spatiales (split_coord) et fusion via split()
        split_pts = PointArray([(2.0, 5.0), (5.0, 13.0)])
        split_bspline = sample_spline.split(split_coord=split_pts)
        assert isinstance(split_bspline, BSpline)
        assert len(split_bspline) == 4
        assert split_bspline[0].p1 == split_bspline[1].p0 == Point(0.0, 5.0)
        assert split_bspline[2].p1 == split_bspline[3].p0 == Point(5.0, 10.0)
        assert split_bspline.arc_length() == pytest.approx(sample_spline.arc_length(), rel=1e-2)

        # 3. Découpe à la jonction exacte entre c1 et c2 (u = 1.0)
        parts_junction = sample_spline.split_into_multiple_bsplines(u=np.array([1.0]))
        assert len(parts_junction) == 2
        assert len(parts_junction[0]) == 1 and parts_junction[0][0].p1 == Point(0.0, 10.0)
        assert len(parts_junction[1]) == 1 and parts_junction[1][0].p0 == Point(0.0, 10.0)

        # 4. Erreur si ni split_coord ni u n'est fourni
        with pytest.raises(ValueError):
            sample_spline.split_into_multiple_bsplines()


class TestBezierPydanticIntegration:
    def test_pydantic_validation_and_serialization(self):
        pydantic = pytest.importorskip("pydantic")

        class PathModel(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(validate_assignment=True)

            curve: BezierCubic
            spline: BSpline

        c_data = ((0.0, 0.0), (1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
        s_data = [c_data, ((5.0, 6.0), (7.0, 8.0), (9.0, 10.0), (11.0, 12.0))]

        model = PathModel(curve=c_data, spline=s_data)
        assert isinstance(model.curve, BezierCubic)
        assert isinstance(model.spline, BSpline)
        assert len(model.spline) == 2

        dumped = model.model_dump(mode="python")
        assert dumped["curve"] == c_data
        assert dumped["spline"] == s_data

        reloaded = PathModel.model_validate_json(model.model_dump_json())
        np.testing.assert_array_equal(reloaded.curve, model.curve)
        np.testing.assert_array_equal(reloaded.spline, model.spline)

        model.curve = ((0.0, 0.0), (1.0, 2.0), (3.0, 4.0), (5.0, 6.0))  # type: ignore
        model.spline = [c_data, ((5.0, 6.0), (7.0, 8.0), (9.0, 10.0), (11.0, 12.0))]  # type: ignore
        assert isinstance(model.curve, BezierCubic)
        assert isinstance(model.spline, BSpline)

        val_schema = PathModel.model_json_schema(mode="validation")
        ser_schema = PathModel.model_json_schema(mode="serialization")
        assert set(val_schema["properties"].keys()) == {"curve", "spline"}
        assert set(ser_schema["properties"].keys()) == {"curve", "spline"}
