from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from glakit.geometry import (
    IntPoint,
    IntPointArray,
    IntPointMap,
    IntRect,
    Point,
    PointArray,
    PointMap,
    Rect,
    VectorMap,
    as_int_point_map,
    as_point_map,
    as_vector_map,
    crop_pad_slices,
)

# =====================================================================
# Tests for Point and IntPoint
# =====================================================================


class TestPoint:
    def test_creation_and_properties(self):
        p1 = Point(1.5, 2.5)
        assert isinstance(p1, Point)
        assert p1.y == 1.5
        assert p1.x == 2.5
        assert p1.tuple == (1.5, 2.5)
        assert p1.xy == (2.5, 1.5)
        assert p1[0] == 1.5
        assert p1[1] == 2.5
        assert str(p1) == "(1.5, 2.5)"

    def test_creation_from_pointlike(self):
        assert Point((1, 2)) == Point(1.0, 2.0)
        assert Point("(3.5, 4.5)") == Point(3.5, 4.5)
        assert Point(np.array([5, 6])) == Point(5.0, 6.0)
        assert Point(7) == Point(7.0, 7.0)

        with pytest.raises(TypeError):
            Point((1, 2), 3)  # type: ignore
        with pytest.raises(TypeError):
            Point("invalid")
        with pytest.raises(TypeError):
            Point("(a, b)")
        with pytest.raises(TypeError):
            Point((1j, 0))  # type: ignore

    def test_immutability_and_hash(self):
        p = Point(1.0, 2.0)
        with pytest.raises(ValueError):
            p[0] = 10.0  # type: ignore
        with pytest.raises(TypeError):
            p += (1.0, 1.0)  # type: ignore

        s = {Point(1.0, 2.0), Point(1.0, 2.0), Point(3.0, 4.0)}
        assert len(s) == 2
        assert Point(1.0, 2.0) in s

    def test_copy_and_equality(self):
        p1 = Point(1.5, 2.5)
        p2 = Point(p1)
        assert p1 == p2
        assert p1 is not p2

        p3 = p2.copy()
        assert isinstance(p3, np.ndarray) and not isinstance(p3, Point)
        assert p1 == p3

        p4 = Point(1.5, 2.6)
        assert p1 != p4

    def test_arithmetic_and_ufuncs(self):
        p = Point(2.0, 1.0)
        res_tuple = p + (3, 4)
        assert isinstance(res_tuple, Point)
        assert res_tuple == Point(5.0, 5.0)

        res_rtuple = (3, 4) + p
        assert isinstance(res_rtuple, Point)
        assert res_rtuple == Point(5.0, 5.0)

        res_arr = p + np.array([2, 1])
        assert isinstance(res_arr, Point)
        assert res_arr == Point(4.0, 2.0)

        res_str = p + "(1.5, 2.5)"
        assert isinstance(res_str, Point)
        assert res_str == Point(3.5, 3.5)

        assert (p * 3) == Point(6.0, 3.0)
        assert (-p) == Point(-2.0, -1.0)

        # Slicing should downgrade to 1D ndarray
        sl = p[:1]
        assert not isinstance(sl, Point)
        assert isinstance(sl, np.ndarray)
        assert sl.shape == (1,)

    def test_geometry_methods(self):
        p = Point(3.0, 4.0)
        assert p.norm == 5.0
        assert math.isclose(p.normalized().norm, 1.0)
        assert Point.origin().normalized() == Point(0.0, 0.0)

        assert Point(1.0, 0.0).rot90() == Point(0.0, 1.0)
        assert Point(1.0, 0.0).rotate(90, mode="deg") == Point(0.0, 1.0)
        assert Point(1.0, 0.0).rotate(math.pi / 2, mode="rad") == Point(0.0, 1.0)

        assert Point(2.0, 3.0).cross((4.0, 5.0)) == 3.0 * 4.0 - 2.0 * 5.0
        assert Point(2.0, 3.0).point_dot((4.0, 5.0)) == 3.0 * 5.0 + 2.0 * 4.0

        assert Point(1.2, 2.8).floor() == IntPoint(1, 2)
        assert Point(1.2, 2.8).ceil() == IntPoint(2, 3)
        assert Point(1.2, 2.8).as_int() == IntPoint(1, 3)

        clipped = Point(15.0, -5.0).clip_to(Rect((0, 0), (10, 10)))
        assert clipped == Point(10.0, 0.0)

    def test_distance(self):
        p = Point(0.0, 0.0)
        d_single = p.distance((3.0, 4.0))
        assert isinstance(d_single, float)
        assert d_single == 5.0

        pts = PointArray([(3.0, 4.0), (6.0, 8.0)])
        d_arr = p.distance(pts)
        assert isinstance(d_arr, np.ndarray)
        np.testing.assert_allclose(d_arr, [5.0, 10.0])


class TestIntPoint:
    def test_creation_and_promotion(self):
        ip = IntPoint(2, 3)
        assert isinstance(ip, IntPoint)
        assert ip.dtype == np.int64
        assert ip.y == 2
        assert ip.x == 3
        assert isinstance(ip.y, int)

        # Int + Int -> IntPoint
        res_int = ip + (4, 5)
        assert isinstance(res_int, IntPoint)
        assert res_int == IntPoint(6, 8)

        # Int + Float -> Point
        res_float = ip + (0.5, 1.5)
        assert isinstance(res_float, Point)
        assert res_float == Point(2.5, 4.5)

        # True division -> Point
        res_div = ip / 2
        assert isinstance(res_div, Point)
        assert res_div == Point(1.0, 1.5)

        # Floor division with int -> IntPoint
        res_fdiv = IntPoint(5, 7) // 2
        assert isinstance(res_fdiv, IntPoint)
        assert res_fdiv == IntPoint(2, 3)

        assert ip.rot90() == IntPoint(-3, 2)
        assert isinstance(ip.rot90(), IntPoint)

    def test_creation_intpointlike(self):
        ip1 = IntPoint((1, 2))
        assert isinstance(ip1, IntPoint)
        assert ip1.y == 1
        assert ip1.x == 2

        ip2 = IntPoint("(3, 4)")
        assert isinstance(ip2, IntPoint)
        assert ip2.y == 3
        assert ip2.x == 4

        ip3 = IntPoint(np.array([5, 6]))
        assert isinstance(ip3, IntPoint)
        assert ip3.y == 5
        assert ip3.x == 6

        with pytest.raises(TypeError):
            IntPoint((1.5, 2.5))
        with pytest.raises(TypeError):
            IntPoint("(1.5, 2.5)")


# =====================================================================
# Tests for PointArray and IntPointArray
# =====================================================================


class TestPointArray:
    def test_creation_indexing_and_mutation(self):
        pts = PointArray([(1, 2), "(3.5, 4.5)", Point(5, 6)])
        assert isinstance(pts, PointArray)
        assert pts.shape == (3, 2)
        np.testing.assert_allclose(pts.y, [1.0, 3.5, 5.0])
        np.testing.assert_allclose(pts.x, [2.0, 4.5, 6.0])

        # Row indexing yields immutable Point
        p0 = pts[0]
        assert isinstance(p0, Point)
        assert p0 == Point(1.0, 2.0)

        # Slicing yields PointArray
        sub = pts[1:]
        assert isinstance(sub, PointArray)
        assert sub.shape == (2, 2)

        # Iteration yields Points
        items = list(pts)
        assert len(items) == 3
        assert all(isinstance(it, Point) for it in items)

        # PointArray is mutable
        pts[0] = "(9, 10)"
        assert pts[0] == Point(9.0, 10.0)

        # Invalid creation and mutation
        with pytest.raises(TypeError):
            PointArray(["(a, 2)", (3.5j, 4.5)])
        with pytest.raises(TypeError):
            pts[0] = (1.5j, 2.5)  # type: ignore

    def test_copy_and_equality(self):
        pts1 = PointArray([(1, 2), (3, 4)])
        pts2 = PointArray(pts1)
        assert np.all(pts1 == pts2)
        assert pts1 is not pts2

        pts3 = pts2.copy()
        assert isinstance(pts3, PointArray)
        assert np.all(pts1 == pts3)
        assert pts1 is not pts3

        pts4 = PointArray([(1, 2), (3, 5)])
        assert np.any(pts1 != pts4)

    def test_vectorized_ops_and_distance(self):
        pts = PointArray([(0, 0), (3, 4)])
        shifted = pts + (1, 2)
        assert isinstance(shifted, PointArray)
        assert shifted[1] == Point(4.0, 6.0)

        d1 = pts.distance((3, 4))
        np.testing.assert_allclose(d1, [5.0, 0.0])

        d2 = pts.distance(PointArray([(0, 0), (6, 8)]))
        assert d2.shape == (2, 2)
        np.testing.assert_allclose(d2, [[0.0, 10.0], [5.0, 5.0]])

    def test_promotion(self):
        p = Point(1.0, 2.0)
        arr = PointArray([[1.0, 2.0], [3.0, 4.0]])
        assert isinstance(p + arr, PointArray)
        assert isinstance(arr + p, PointArray)


class TestIntPointArray:
    def test_creation_and_promotion(self):
        ipts = IntPointArray([(1, 2), (3, 4)])
        assert isinstance(ipts, IntPointArray)
        assert ipts.dtype == np.int64
        assert isinstance(ipts[0], IntPoint)

        # Invalid creation and mutation
        with pytest.raises(TypeError):
            IntPointArray([(1, 2), (3.5, 4.5)])
        with pytest.raises(TypeError):
            ipts[0] = (1.5, 2.5)  # type: ignore
        with pytest.raises(TypeError):
            ipts[0] = Point(1.5, 2.5)  # type: ignore

        # Int + Int -> IntPointArray
        res_int = ipts + (10, 20)
        assert isinstance(res_int, IntPointArray)
        assert res_int[0] == IntPoint(11, 22)

        # Int + Float -> PointArray
        res_float = ipts + (0.5, 0.5)
        assert isinstance(res_float, PointArray)
        assert res_float[0] == Point(1.5, 2.5)


# =====================================================================
# Tests for Rect and IntRect
# =====================================================================
class TestPointMaps:
    # =========================================================================
    # 1. Construction, dtypes & zéro-copie
    # =========================================================================
    def test_zeros_and_dtypes(self) -> None:
        pm = PointMap.zeros((3, 4))
        ipm = IntPointMap.zeros((3, 4))
        vm = VectorMap.zeros((3, 4))

        assert pm.shape == (3, 4, 2) and pm.dtype == np.float64
        assert ipm.shape == (3, 4, 2) and ipm.dtype == np.int64
        assert vm.shape == (3, 4, 2) and vm.dtype == np.float32

    def test_as_map_zero_copy_when_matching_dtype(self) -> None:
        arr_f32 = np.ones((2, 3, 2), dtype=np.float32)
        arr_f64 = np.ones((2, 3, 2), dtype=np.float64)
        arr_i64 = np.ones((2, 3, 2), dtype=np.int64)

        vm = as_vector_map(arr_f32, copy=False)
        pm = as_point_map(arr_f64, copy=False)
        ipm = as_int_point_map(arr_i64, copy=False)

        assert isinstance(vm, VectorMap) and np.shares_memory(vm, arr_f32)
        assert isinstance(pm, PointMap) and np.shares_memory(pm, arr_f64)
        assert isinstance(ipm, IntPointMap) and np.shares_memory(ipm, arr_i64)

    def test_construction_from_nested_sequences(self) -> None:
        data = [[(0.5, 1.5), (2.0, 3.0)], [(4.5, 5.5), (6.0, 7.0)]]
        vm = VectorMap(data)
        pm = PointMap(data)

        assert isinstance(vm, VectorMap) and vm.dtype == np.float32
        assert isinstance(pm, PointMap) and pm.dtype == np.float64
        np.testing.assert_allclose(vm.ndarray, pm.ndarray)

    def test_int_point_map_rejects_fractional_floats(self) -> None:
        with pytest.raises(TypeError):
            IntPointMap([[(0.5, 1.5)]])

        # Des floats à valeur entière doivent en revanche être acceptés sans perte
        ipm = IntPointMap([[(1.0, 2.0), (3.0, 4.0)]])
        assert isinstance(ipm, IntPointMap) and ipm.dtype == np.int64

    def test_cross_conversions(self) -> None:
        pm = PointMap([[(1.2, 2.7), (-3.6, 4.1)]])

        vm = pm.as_vector()
        assert isinstance(vm, VectorMap) and vm.dtype == np.float32
        np.testing.assert_allclose(vm.ndarray, pm.ndarray, rtol=1e-6)

        ipm = vm.as_int()
        assert isinstance(ipm, IntPointMap) and ipm.dtype == np.int64
        np.testing.assert_array_equal(ipm.ndarray, np.array([[[1, 3], [-4, 4]]]))

        pm_back = ipm.as_float()
        assert isinstance(pm_back, PointMap) and pm_back.dtype == np.float64

    # =========================================================================
    # 2. Indexation, slicing, assignation & propriétés
    # =========================================================================
    def test_indexing_and_slicing(self) -> None:
        vm = VectorMap([[(1.0, 2.0), (3.0, 4.0)], [(5.0, 6.0), (7.0, 8.0)]])
        ipm = IntPointMap([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])

        # Accès à un élément (y, x) -> Point / IntPoint
        pt = vm[0, 1]
        assert isinstance(pt, Point) and pt == Point(3.0, 4.0)
        ipt = ipm[0, 1]
        assert isinstance(ipt, IntPoint) and ipt == IntPoint(3, 4)

        # Accès à une ligne -> PointArray / IntPointArray
        row = vm[1]
        assert isinstance(row, PointArray) and row.shape == (2, 2)
        irow = ipm[1]
        assert isinstance(irow, IntPointArray) and irow.shape == (2, 2)

        # Slicing 2D -> conserve le type de Map
        sub_vm = vm[:, :1]
        assert isinstance(sub_vm, VectorMap) and sub_vm.shape == (2, 1, 2) and sub_vm.dtype == np.float32

        # Extraction d'une composante -> ndarray brut 2D
        assert type(vm.y) is np.ndarray and vm.y.dtype == np.float32
        assert type(vm.x) is np.ndarray and vm.x.dtype == np.float32
        assert type(vm[..., 0]) is np.ndarray

    def test_item_assignment(self) -> None:
        vm = VectorMap.zeros((2, 2))
        vm[0, 0] = (1.5, 2.5)
        vm[1] = [(3.0, 4.0), (5.0, 6.0)]
        assert vm.dtype == np.float32
        assert vm[0, 0] == Point(1.5, 2.5)
        assert vm[1, 1] == Point(5.0, 6.0)

        ipm = IntPointMap.zeros((2, 2))
        ipm[0, 0] = (2, 3)
        assert ipm[0, 0] == IntPoint(2, 3)

        with pytest.raises(TypeError):
            ipm[0, 1] = (1.5, 2.5)  # type: ignore

    def test_flatten_as_array(self) -> None:
        vm = VectorMap.zeros((3, 4))
        flat = vm.as_array()
        assert isinstance(flat, PointArray)
        assert flat.shape == (12, 2)

    # =========================================================================
    # 3. Règles de promotion arithmétique (IntPointMap < VectorMap < PointMap)
    # =========================================================================
    def test_vector_map_preserves_float32_with_scalars_and_tuples(self) -> None:
        vm = VectorMap([[(1.0, 2.0), (3.0, 4.0)]])

        # Scalaires Python (int & float) -> doit rester VectorMap (float32)
        res_mul_int = vm * 2
        res_mul_float = vm * 0.5
        res_div = vm / 2
        res_add_scalar = vm + 1.5
        res_rsub = 10.0 - vm

        for res in (res_mul_int, res_mul_float, res_div, res_add_scalar, res_rsub):
            assert isinstance(res, VectorMap)
            assert res.dtype == np.float32

        # Tuples/listes de littéraux Python -> doit rester VectorMap (float32)
        res_tuple = vm + (0.5, 1.5)
        res_list = vm - [(0.5, 0.5), (1.5, 1.5)]
        assert isinstance(res_tuple, VectorMap) and res_tuple.dtype == np.float32
        assert isinstance(res_list, VectorMap) and res_list.dtype == np.float32
        np.testing.assert_allclose(res_tuple.ndarray, np.array([[[1.5, 3.5], [3.5, 5.5]]], dtype=np.float32))

    def test_arithmetic_promotion_hierarchy(self) -> None:
        ipm = IntPointMap([[(1, 2), (3, 4)]])
        vm = VectorMap([[(0.5, 0.5), (1.5, 1.5)]])
        pm = PointMap([[(10.0, 20.0), (30.0, 40.0)]])

        # IntPointMap + IntPointMap -> IntPointMap
        r_ii = ipm + ipm
        assert isinstance(r_ii, IntPointMap) and r_ii.dtype == np.int64

        # IntPointMap + (0.5, 1.5) -> PointMap (float64)
        r_if = ipm + (0.5, 1.5)
        assert isinstance(r_if, PointMap) and r_if.dtype == np.float64

        # VectorMap + IntPointMap -> PointMap (float32)
        r_vi = vm + ipm
        assert isinstance(r_vi, PointMap) and r_vi.dtype == np.float64

        # VectorMap + VectorMap -> VectorMap (float32)
        r_vv = vm + vm
        assert isinstance(r_vv, VectorMap) and r_vv.dtype == np.float32

        # VectorMap + Point (explicit float64) -> PointMap (float64)
        r_vpt = vm + Point(1.0, 2.0)
        assert isinstance(r_vpt, PointMap) and r_vpt.dtype == np.float64

        # VectorMap + PointMap -> PointMap (float64)
        r_vp = vm + pm
        assert isinstance(r_vp, PointMap) and r_vp.dtype == np.float64

        # Réduction sur un axe (axis=0) -> PointArray
        col_mean = np.mean(vm, axis=0)
        assert isinstance(col_mean, PointArray)

    # =========================================================================
    # 4. Opérations géométriques et vectorielles
    # =========================================================================
    def test_norm_angle_and_normalized(self) -> None:
        # Vecteurs (y, x): (3, 4) -> norme 5 ; (0, 0) -> vecteur nul ; (0, 2) -> angle 0
        vm = VectorMap([[(3.0, 4.0), (0.0, 0.0), (0.0, 2.0)]])

        n = vm.norm
        assert n.dtype == np.float32
        np.testing.assert_allclose(n, np.array([[5.0, 0.0, 2.0]], dtype=np.float32))

        ang = vm.angle()
        assert ang.dtype == np.float32
        np.testing.assert_allclose(ang[0, 2], 0.0)

        normed = vm.normalized()
        assert isinstance(normed, VectorMap) and normed.dtype == np.float32
        np.testing.assert_allclose(normed.ndarray, np.array([[[0.6, 0.8], [0.0, 0.0], [0.0, 1.0]]], dtype=np.float32))

        # Vérifie que PointMap.normalized() retourne bien un PointMap (float64)
        pm_normed = vm.as_float().normalized()
        assert isinstance(pm_normed, PointMap) and pm_normed.dtype == np.float64

    def test_dot_and_cross_products(self) -> None:
        # Vecteurs (y, x)
        v1 = VectorMap([[(0.0, 1.0), (1.0, 0.0)]])  # (y=0, x=1) et (y=1, x=0)
        v2 = VectorMap([[(1.0, 0.0), (1.0, 0.0)]])  # (y=1, x=0) et (y=1, x=0)

        # Produit scalaire : (0*1 + 1*0) = 0 ; (1*1 + 0*0) = 1
        dot = v1.dot_prod(v2)
        assert dot.dtype == np.float32
        np.testing.assert_allclose(dot, np.array([[0.0, 1.0]], dtype=np.float32))

        # Produit vectoriel 2D (x1*y2 - y1*x2) : (1*1 - 0*0) = 1 ; (0*1 - 1*0) = 0
        cross = v1.cross(v2)
        assert cross.dtype == np.float32
        np.testing.assert_allclose(cross, np.array([[1.0, 0.0]], dtype=np.float32))

    # =========================================================================
    # 5. Intégration Pydantic
    # =========================================================================
    def test_pydantic_validation_and_serialization(self) -> None:
        class FlowModel(BaseModel):
            vectors: VectorMap[Literal[2], Literal[2]]
            points: PointMap[Literal[2], Literal[2]]
            grid: IntPointMap[Literal[2], Literal[2]]

        payload = {
            "vectors": [[(0.5, 1.5), (2.0, 3.0)], [(4.0, 5.0), (6.0, 7.0)]],
            "points": [[(0.5, 1.5), (2.0, 3.0)], [(4.0, 5.0), (6.0, 7.0)]],
            "grid": [[(1, 2), (3, 4)], [(5, 6), (7, 8)]],
        }

        model = FlowModel.model_validate(payload)
        assert isinstance(model.vectors, VectorMap) and model.vectors.dtype == np.float32
        assert isinstance(model.points, PointMap) and model.points.dtype == np.float64
        assert isinstance(model.grid, IntPointMap) and model.grid.dtype == np.int64

        # Test serialization
        dumped = model.model_dump()
        assert dumped["grid"] == [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
        assert dumped["vectors"] == [[(0.5, 1.5), (2.0, 3.0)], [(4.0, 5.0), (6.0, 7.0)]]

        # Validation échoue si des floats fractionnaires sont passés à IntPointMap
        with pytest.raises(ValidationError):
            FlowModel.model_validate({**payload, "grid": [[(1.5, 2.0), (3, 4)], [(5, 6), (7, 8)]]})


# =====================================================================
# Tests for Rect and IntRect
# =====================================================================


class TestRect:
    def test_creation_and_normalization(self):
        r = Rect((10, 20), (0, 5))
        assert isinstance(r, Rect)
        assert r.p0 == Point(10.0, 20.0)
        assert r.p1 == Point(0.0, 5.0)
        assert r.center == Point(5.0, 12.5)
        assert r.signed_sizes == Point(-10.0, -15.0)
        assert r.signed_h == -10.0
        assert r.signed_w == -15.0

        # Ensure top_left and bottom_right ensure positive sizes even if p0 and p1 are not ordered
        assert r.top_left == Point(0.0, 5.0)
        assert r.bottom_right == Point(10.0, 20.0)
        assert r.top_right == Point(0.0, 20.0)
        assert r.bottom_left == Point(10.0, 5.0)
        assert r.top == 0.0
        assert r.left == 5.0
        assert r.bottom == 10.0
        assert r.right == 20.0
        assert r.h == 10.0
        assert r.w == 15.0
        assert r.area == 150.0

        # Single PointLike creates Rect from origin (0, 0)
        r_single = Rect((10, 20))
        assert r_single == Rect((0, 0), (10, 20))

        # Flat 4-tuple
        assert Rect((1, 2, 3, 4)) == Rect((1, 2), (3, 4))

        # from_center
        rc = Rect.from_center((10, 20), (4, 6))
        assert rc == Rect((8, 17), (12, 23))

    def test_immutability_and_indexing(self):
        r = Rect((0, 0), (10, 20))
        with pytest.raises(TypeError):
            r[0] = (1, 1)  # type: ignore

        assert r[0] == Point(0.0, 0.0)
        assert r[1] == Point(10.0, 20.0)
        assert isinstance(r[0], Point)

        corners = r.corners()
        assert isinstance(corners, PointArray)
        assert corners.shape == (4, 2)

    def test_copy_and_equality(self):
        r1 = Rect((0, 0), (10, 20))
        r2 = Rect(r1)
        assert r1 == r2
        assert r1 is not r2

        r3 = r2.copy()
        assert isinstance(r3, np.ndarray) and not isinstance(r3, Rect)
        assert r1 == r3

        r4 = Rect((0, 0), (10, 21))
        assert r1 != r4

    def test_set_operations_and_contains(self):
        r1 = Rect((0, 0), (10, 10))
        r2 = Rect((5, 5), (15, 15))

        inter = r1 & r2
        assert isinstance(inter, Rect)
        assert inter == Rect((5, 5), (10, 10))

        uni = r1 | r2
        assert isinstance(uni, Rect)
        assert uni == Rect((0, 0), (15, 15))

        # Union with empty Rect should return the other Rect
        assert uni | Rect.empty() is uni

        # Disjoint intersection -> empty
        disjoint = r1 & Rect((20, 20), (30, 30))
        assert disjoint.is_empty()

        assert r1.contains((5, 5))
        assert not r1.contains((12, 5))
        assert r1.contains(Rect((2, 2), (8, 8)))
        assert not r1.contains(r2)

    def test_pad_crop_and_bounding_box(self):
        r = Rect((10, 20), (30, 40))
        assert r.pad(5) == Rect((5, 15), (35, 45))
        assert r.pad(2, 4) == Rect((8, 16), (32, 44))
        assert r.pad(1, 2, 3, 4) == Rect((9, 18), (33, 44))

        pts = PointArray([(5, 25), (15, 50), (35, 30)])
        cropped = r.crop(pts)
        assert cropped[0] == Point(10.0, 25.0)
        assert cropped[1] == Point(15.0, 40.0)
        assert cropped[2] == Point(30.0, 30.0)

        bbox = Rect.bounding_box(pts)
        assert bbox == Rect((5, 25), (35, 50))


class TestIntRect:
    def test_operations_and_promotion(self):
        ir1 = IntRect((0, 0), (10, 10))
        ir2 = IntRect((5, 5), (15, 15))

        assert isinstance(ir1, IntRect)
        assert ir1.dtype == np.int64
        assert isinstance(ir1.top_left, IntPoint)

        # IntRect + IntTuple -> IntRect
        shifted = ir1 + (2, 3)
        assert isinstance(shifted, IntRect)
        assert shifted == IntRect((2, 3), (12, 13))

        # IntRect + FloatTuple -> Rect
        shifted_f = ir1 + (0.5, 0.5)
        assert isinstance(shifted_f, Rect)
        assert shifted_f == Rect((0.5, 0.5), (10.5, 10.5))

        # IntRect / Int -> Rect
        divided = ir1 / 2
        assert isinstance(divided, Rect)
        assert divided == Rect((0.0, 0.0), (5.0, 5.0))

        # IntRect // Int -> IntRect
        fdivided = ir1 // 3
        assert isinstance(fdivided, IntRect)
        assert fdivided == IntRect((0, 0), (3, 3))

        # IntRect & IntRect -> IntRect
        inter_int = ir1 & ir2
        assert isinstance(inter_int, IntRect)
        assert inter_int == IntRect((5, 5), (10, 10))

        # IntRect & Rect -> Rect
        inter_mix = ir1 & Rect((2.5, 2.5), (8.5, 8.5))
        assert isinstance(inter_mix, Rect)
        assert inter_mix == Rect((2.5, 2.5), (8.5, 8.5))

        # Disjoint IntRect intersection -> IntRect.empty()
        empty_inter = ir1 & IntRect((20, 20), (30, 30))
        assert isinstance(empty_inter, IntRect)
        assert empty_inter.is_empty()

    def test_image_slicing_and_crop_pad(self):
        img = np.arange(100, dtype=np.int32).reshape(10, 10)
        roi = IntRect((2, 3), (6, 8))

        # Direct slice
        sl = roi.slice()
        np.testing.assert_array_equal(img[sl], img[2:6, 3:8])

        # crop_pad_image with out-of-bounds padding
        padded_roi = IntRect((-2, -1), (4, 5))
        cropped_padded = padded_roi.crop_pad_image(img)
        assert cropped_padded.shape == (6, 6)
        # Top 2 rows and left 1 col should be zero-padded
        assert np.all(cropped_padded[:2, :] == 0)
        assert np.all(cropped_padded[:, :1] == 0)
        np.testing.assert_array_equal(cropped_padded[2:, 1:], img[0:4, 0:5])

        # Channel-last 3D image support
        img_hwc = np.ones((10, 10, 3), dtype=np.uint8)
        out_hwc = padded_roi.crop_pad_image(img_hwc, channel_last=True)
        assert out_hwc.shape == (6, 6, 3)
        assert np.all(out_hwc[:2, :, :] == 0)
        assert np.all(out_hwc[2:, 1:, :] == 1)

    def test_crop_pad_slices_helper(self):
        src = IntRect((0, 0), (10, 10))
        dst = IntRect((5, 5), (15, 15))
        src_sl, dst_sl = crop_pad_slices(src, dst, prefix_ellipsis=False)
        assert src_sl == (slice(5, 10), slice(5, 10))
        assert dst_sl == (slice(0, 5), slice(0, 5))

        src_el, dst_el = crop_pad_slices(src, dst, prefix_ellipsis=True)
        assert src_el == (..., slice(5, 10), slice(5, 10))
        assert dst_el == (..., slice(0, 5), slice(0, 5))


class TestPydanticIntegration:
    def test_pydantic_validation_and_serialization(self):
        pydantic = pytest.importorskip("pydantic")

        class SceneAnnotation(pydantic.BaseModel):
            model_config = pydantic.ConfigDict(validate_assignment=True)
            pt: Point
            ipt: IntPoint
            pts: PointArray
            ipts: IntPointArray
            rect: Rect
            irect: IntRect

        ann = SceneAnnotation(
            pt="(1.5, 2.5)",  # type: ignore
            ipt=(3, 4),  # type: ignore
            pts=[(0.0, 1.0), (2.0, 3.0)],  # type: ignore
            ipts=[(10, 20), (30, 40)],  # type: ignore
            rect=(0.5, 1.5, 10.5, 20.5),  # type: ignore
            irect=(0, 0, 100, 200),  # type: ignore
        )

        assert isinstance(ann.pt, Point) and ann.pt == Point(1.5, 2.5)
        assert isinstance(ann.ipt, IntPoint) and ann.ipt == IntPoint(3, 4)
        assert isinstance(ann.pts, PointArray) and ann.pts.shape == (2, 2)
        assert isinstance(ann.ipts, IntPointArray) and ann.ipts.shape == (2, 2)
        assert isinstance(ann.rect, Rect) and ann.rect == Rect((0.5, 1.5), (10.5, 20.5))
        assert isinstance(ann.irect, IntRect) and ann.irect == IntRect((0, 0), (100, 200))

        # Model serialization
        data = ann.model_dump(mode="python")
        assert data["pt"] == (1.5, 2.5)
        assert data["ipt"] == (3, 4)
        assert data["pts"] == [(0.0, 1.0), (2.0, 3.0)]
        assert data["ipts"] == [(10, 20), (30, 40)]
        assert data["rect"] == ((0.5, 1.5), (10.5, 20.5))
        assert data["irect"] == ((0, 0), (100, 200))

        # JSON Schema generation (validation and serialization modes)
        val_schema = SceneAnnotation.model_json_schema(mode="validation")
        ser_schema = SceneAnnotation.model_json_schema(mode="serialization")

        for schema in (val_schema, ser_schema):
            assert set(schema["properties"].keys()) == {"pt", "ipt", "pts", "ipts", "rect", "irect"}

        # Validation mode exposes union ("anyOf") for Point/Rect and "array" for PointArray
        assert "anyOf" in val_schema["properties"]["pt"]
        assert "anyOf" in val_schema["properties"]["rect"]
        assert val_schema["properties"]["pts"]["type"] == "array"

        # Serialization mode exposes fixed-length arrays ("type": "array")
        ser_props = ser_schema["properties"]
        assert ser_props["pt"]["type"] == "array"
        assert ser_props["pt"]["minItems"] == 2 and ser_props["pt"]["maxItems"] == 2
        assert ser_props["rect"]["type"] == "array"
        assert ser_props["rect"]["minItems"] == 2 and ser_props["rect"]["maxItems"] == 2
        assert ser_props["pts"]["type"] == "array"

        # Round-trip JSON serialization
        json_data = ann.model_dump_json()
        reloaded = SceneAnnotation.model_validate_json(json_data)
        assert reloaded.pt == ann.pt
        assert reloaded.ipt == ann.ipt
        np.testing.assert_allclose(reloaded.pts, ann.pts)
        np.testing.assert_array_equal(reloaded.ipts, ann.ipts)
        assert reloaded.rect == ann.rect
        assert reloaded.irect == ann.irect

        # Validation error on invalid inputs
        with pytest.raises(pydantic.ValidationError):
            SceneAnnotation(
                pt="bad_point",  # type: ignore
                ipt=(3, 4),
                pts=[(0, 1)],
                ipts=[(1, 2)],
                rect=(0, 0, 1, 1),
                irect=(0, 0, 1, 1),
            )

        with pytest.raises(pydantic.ValidationError):
            ann.ipts = (1.5, 2.5)  # type: ignore
