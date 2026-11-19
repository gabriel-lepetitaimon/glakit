from __future__ import annotations

import math

import numpy as np
import pytest

from glakit.geometry import (
    IntPoint,
    IntPointArray,
    IntRect,
    Point,
    PointArray,
    Rect,
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
        p + np.array([[1.0, 2.0], [3.0, 4.0]])
        arr = PointArray([p])
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
