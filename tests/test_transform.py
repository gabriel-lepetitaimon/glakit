from __future__ import annotations

import numpy as np
import pytest
from pydantic import BaseModel

from glakit.geometry import IntRect, Point, PointArray
from glakit.transform import (
    AffineTransform,
    FlipTransform,
    IdentityTransform,
    InverseTransform,
    ResizeTranslation,
    SimilarityTransform,
    Transform,
    TransformComposition,
    Translation,
)


@pytest.fixture
def sample_points() -> PointArray:
    return PointArray([(0.0, 0.0), (10.0, 0.0), (0.0, 20.0), (10.0, 20.0)])


# ==========================================
# Tests IdentityTransform
# ==========================================
def test_identity_transform(sample_points: PointArray) -> None:
    T = Transform.identity()
    assert isinstance(T, IdentityTransform)
    assert T.is_identity()
    assert T.is_exact
    assert T.is_inverse_exact
    assert T.is_yx_independent
    assert T.invert().is_identity()
    assert T.simplify() is T

    out = T(sample_points)
    assert isinstance(out, PointArray)
    assert np.allclose(out, sample_points)

    pt = Point(3.5, -2.0)
    out_pt = T(pt)
    assert isinstance(out_pt, Point)
    assert out_pt == pt

    assert np.allclose(T.transform_inverse(sample_points), sample_points)


# ==========================================
# Tests Translation
# ==========================================
class TestTranslation:
    def test_transform(self, sample_points: PointArray) -> None:
        t = Point(5.0, -3.0)
        T = Translation(t)
        assert not T.is_identity()
        assert Translation((0, 0)).is_identity()
        assert T.is_yx_independent

        # Transformation d'un point et d'un tableau de points
        assert T(Point(1.0, 2.0)) == Point(6.0, -1.0)
        transformed = T(sample_points)
        assert isinstance(transformed, PointArray)
        assert np.allclose(transformed, sample_points + t)

        # Inverse
        T_inv = T.invert()
        assert isinstance(T_inv, Translation)
        assert np.allclose(T_inv(transformed), sample_points)
        assert np.allclose(T.transform_inverse(transformed), sample_points)
        assert (T @ T_inv).is_identity()

        # Composition
        T2 = Translation((2.0, 4.0))
        composed = T @ T2
        assert isinstance(composed, Translation)
        assert composed.t == Point(7.0, 1.0)

    def test_fit(self, sample_points: PointArray) -> None:
        t_true = Point(-4.5, 12.2)
        dst = sample_points + t_true
        T_fit, mse = Translation.fit(sample_points, dst)
        assert isinstance(T_fit, Translation)
        assert T_fit.t == t_true
        assert np.isclose(mse, 0.0)


# ==========================================
# Tests ResizeTranslation
# ==========================================
class TestResizeTranslation:
    def test_transform(self, sample_points: PointArray) -> None:
        T = ResizeTranslation(s=(2.0, 3.0), t=(10.0, -5.0))
        assert not T.is_identity()
        assert ResizeTranslation(s=1.0, t=(0.0, 0.0)).is_identity()
        assert T.is_yx_independent

        pt = Point(4.0, 5.0)
        assert T(pt) == Point(2.0 * 4.0 + 10.0, 3.0 * 5.0 - 5.0)

        # Inverse
        T_inv = T.invert()
        assert isinstance(T_inv, ResizeTranslation)
        assert np.allclose(T_inv(T(sample_points)), sample_points)
        assert np.allclose(T.transform_inverse(T(sample_points)), sample_points)
        assert (T @ T_inv).is_identity()

        # Constructeurs alternatifs
        T_tr = ResizeTranslation.translate_resize(t=(2.0, 3.0), s=4.0)
        assert T_tr(Point(1.0, 1.0)) == Point((1.0 + 2.0) * 4.0, (1.0 + 3.0) * 4.0)

        center = Point(10.0, 10.0)
        T_res = ResizeTranslation.resize(s=2.0, center=center)
        assert T_res(center) == center
        assert T_res(Point(12.0, 15.0)) == Point(14.0, 20.0)

    def test_fit(self, sample_points: PointArray) -> None:
        # 1. Échelle uniforme
        T_uni = ResizeTranslation(s=2.5, t=(-3.0, 7.0))
        dst_uni = T_uni(sample_points)
        fitted_uni, mse_uni = ResizeTranslation.fit(sample_points, dst_uni, uniform_scale=True)
        assert isinstance(fitted_uni, ResizeTranslation)
        assert np.allclose(fitted_uni.s, [2.5, 2.5])
        assert np.allclose(fitted_uni.t, [-3.0, 7.0])
        assert np.isclose(mse_uni, 0.0)

        # 2. Échelle non-uniforme (s_y != s_x)
        T_non_uni = ResizeTranslation(s=(1.5, 3.5), t=(4.0, -8.0))
        dst_non_uni = T_non_uni(sample_points)
        fitted_non_uni, mse_non_uni = ResizeTranslation.fit(sample_points, dst_non_uni, uniform_scale=False)
        assert isinstance(fitted_non_uni, ResizeTranslation)
        assert np.allclose(fitted_non_uni.s, [1.5, 3.5])
        assert np.allclose(fitted_non_uni.t, [4.0, -8.0])
        assert np.isclose(mse_non_uni, 0.0)


# ==========================================
# Tests SimilarityTransform
# ==========================================
class TestSimilarityTransform:
    def test_transform(self, sample_points: PointArray) -> None:
        T = SimilarityTransform(s=2.0, r=np.pi / 4, t=(3.0, -2.0))
        assert not T.is_identity()
        assert not T.is_yx_independent
        assert SimilarityTransform(s=1.0, r=0.0, t=(0.0, 0.0)).is_identity()
        assert SimilarityTransform(s=2.0, r=0.0).is_yx_independent

        transformed = T(sample_points)
        T_inv = T.invert()
        assert isinstance(T_inv, SimilarityTransform)
        assert np.allclose(T_inv(transformed), sample_points)
        assert np.allclose(T.transform_inverse(transformed), sample_points)
        assert (T @ T_inv).is_identity()

        # Composition de deux similarités
        T2 = SimilarityTransform(s=0.5, r=-np.pi / 4, t=(1.0, 1.0))
        composed = T @ T2
        assert isinstance(composed, SimilarityTransform)
        assert np.allclose(composed(sample_points), T(T2(sample_points)))

    def test_fit(self, sample_points: PointArray) -> None:
        T_true = SimilarityTransform(s=1.8, r=0.6, t=(-5.0, 9.0))
        dst = T_true(sample_points)
        T_fit, mse = SimilarityTransform.fit(sample_points, dst)
        assert isinstance(T_fit, SimilarityTransform)
        assert np.isclose(T_fit.s, T_true.s)
        assert np.isclose(T_fit.r, T_true.r)
        assert T_fit.t == T_true.t
        assert np.isclose(mse, 0.0)


# ==========================================
# Tests AffineTransform & FlipTransform
# ==========================================
class TestAffineAndFlipTransform:
    def test_flip_transform(self, sample_points: PointArray) -> None:
        center = Point(5.0, 10.0)
        T_h = FlipTransform(center=center, horizontal=True, vertical=False)
        assert T_h(Point(2.0, 4.0)) == Point(2.0, 16.0)
        assert T_h(center) == center
        assert (T_h @ T_h).is_identity()

        T_hv = FlipTransform(center=center, horizontal=True, vertical=True)
        assert T_hv(Point(2.0, 4.0)) == Point(8.0, 16.0)
        assert np.allclose(T_hv.transform_inverse(T_hv(sample_points)), sample_points)

    def test_affine_transform(self, sample_points: PointArray) -> None:
        H = np.array([[1.5, 0.2], [-0.3, 2.0]])
        t = Point(4.0, -1.0)
        T = AffineTransform(H, t)

        transformed = T(sample_points)
        T_inv = T.invert()
        assert np.allclose(T_inv(transformed), sample_points)
        assert (T @ T_inv).is_identity()

        # Rotation autour d'un centre
        center = Point(5.0, 5.0)
        T_rot = AffineTransform.rotate(90.0, center=center)
        assert T_rot(center) == center
        assert np.allclose(T_rot(Point(5.0, 10.0)), Point(10.0, 5.0))

        # Conversions spécialisées
        assert AffineTransform(np.eye(2), (2.0, 3.0)).as_translation() == Translation((2.0, 3.0))
        assert T.as_translation() is None

        sim = SimilarityTransform(s=2.0, r=0.3, t=(1.0, 2.0))
        aff_sim = AffineTransform(sim.H, sim.t)
        assert aff_sim.as_similarity_transform() == sim

        res = ResizeTranslation(s=(2.0, 3.0), t=(1.0, 2.0))
        aff_res = AffineTransform(res.H, res.t)
        assert aff_res.as_resize_translation() == res

    def test_affine_transform_fit(self, sample_points: PointArray) -> None:
        H = np.array([[1.2, -0.4], [0.5, 0.8]])
        t = Point(3.0, -7.0)
        T_true = AffineTransform(H, t)
        dst = T_true(sample_points)

        T_fit, mse = AffineTransform.fit(sample_points, dst)
        assert T_fit == T_true
        assert np.isclose(mse, 0.0)


# ==========================================
# Tests Composition, Inversion & Domaines
# ==========================================
def test_composition_and_simplification(sample_points: PointArray) -> None:
    T1 = Translation((2.0, 3.0))
    T2 = FlipTransform((5.0, 5.0), horizontal=True)
    T3 = ResizeTranslation(s=2.0, t=(1.0, 1.0))

    comp = T1 @ T2 @ T3
    assert isinstance(comp, AffineTransform)  # Composition of AffineTransform subclasses results in AffineTransform
    assert np.allclose(comp(sample_points), T1(T2(T3(sample_points))))

    comp_inv = comp.invert()
    assert np.allclose(comp_inv(comp(sample_points)), sample_points)
    assert (comp @ comp_inv).simplify().is_identity()


def test_fit_to_projection(sample_points: PointArray) -> None:
    T_true = Translation((3.0, -2.0))
    dst = T_true(sample_points)

    models = {2: Translation, 3: SimilarityTransform, 4: AffineTransform}
    fitted, mse = Transform.fit_to_projection(sample_points[:3], dst[:3], models)
    assert isinstance(fitted, SimilarityTransform)
    assert np.isclose(mse, 0.0)

    with pytest.raises(ValueError):
        Transform.fit_to_projection(sample_points[:1], dst[:1], models)


def test_domain_transformations() -> None:
    domain = IntRect((0, 0), (10, 20))
    T = ResizeTranslation(s=2.0, t=(5, 10))
    transformed_domain = T.transform_domain(domain)
    assert isinstance(transformed_domain, IntRect)
    assert transformed_domain == IntRect((5, 10), (25, 50))

    inv_domain = T.inverse_transform_domain(transformed_domain)
    assert inv_domain == domain


# ==========================================
# Tests Warping d'images & Grille
# ==========================================
def test_warp_translation_and_affine() -> None:
    img = np.zeros((10, 12), dtype=np.uint8)
    img[2:5, 3:7] = 255

    # Translation entière exacte
    T_trans = Translation((3, 4))
    warped, w_domain = T_trans.warp(img, src_top_left=(0, 0), warped_domain="full")
    assert w_domain == IntRect((3, 4), (13, 16))
    assert warped.shape == (10, 12)
    np.testing.assert_array_equal(warped, img)

    # Conservation du domaine ("same")
    warped_same, w_domain_same = T_trans.warp(img, warped_domain="same")
    assert w_domain_same == IntRect((0, 0), (10, 12))
    assert np.all(warped_same[5:8, 7:11] == 255)

    # Image booléenne
    bool_img = img > 0
    warped_bool, _ = T_trans.warp(bool_img, warped_domain="full")
    assert warped_bool.dtype == np.bool_
    np.testing.assert_array_equal(warped_bool, bool_img)

    # Image multi-canaux (channel_last=False)
    chw_img = np.stack([img, img], axis=0)
    warped_chw, _ = T_trans.warp(chw_img, channel_last=False)
    assert warped_chw.shape == (2, 10, 12)


def test_draw_grid() -> None:
    domain = IntRect((0, 0), (100, 100))
    T = ResizeTranslation(s=0.5, t=(10, 10))
    grid, grid_domain = T.draw_grid(domain, subdivision=4, subsampling=2)
    assert grid.dtype == np.bool_
    assert grid.any()
    assert isinstance(grid_domain, IntRect)


# ==========================================
# Tests Pydantic Serialization
# ==========================================
class _SingleTransformModel(BaseModel):
    transform: Transform


class _TypedTransformsModel(BaseModel):
    translation: Translation
    similarity: SimilarityTransform
    affine: AffineTransform


class TestTransformPydantic:
    @pytest.mark.parametrize(
        "transform",
        [
            IdentityTransform(),
            Translation((12.5, -4.0)),
            ResizeTranslation((2.0, 0.5), (10.0, 20.0)),
            SimilarityTransform(s=1.5, r=0.35, t=(-3.0, 8.0)),
            FlipTransform(center=(100.0, 200.0), horizontal=True, vertical=False),
            AffineTransform(H=np.array([[1.2, -0.3], [0.4, 0.9]]), t=(5.0, -2.0)),
        ],
    )
    def test_roundtrip_json_polymorphic(self, transform: Transform) -> None:
        model = _SingleTransformModel(transform=transform)
        json_str = model.model_dump_json()

        restored = _SingleTransformModel.model_validate_json(json_str)
        assert type(restored.transform) is type(transform)
        assert restored.transform == transform

    def test_roundtrip_composition_and_inverse(self) -> None:
        class _CustomNonInvertible(Transform):
            def __init__(self, offset: float = 1.0) -> None:
                self.offset = offset

            def transform(self, src):
                return PointArray.as_array(src) + self.offset

            def _to_dict(self):
                return {"offset": self.offset}

        t1 = Translation((10.0, 20.0))
        t2 = SimilarityTransform(s=2.0, r=0.5, t=(1.0, 2.0))
        t3 = InverseTransform(_CustomNonInvertible(3.5))
        comp = TransformComposition(t1, t2, t3, sequential_warp=True)

        model = _SingleTransformModel(transform=comp)
        restored = _SingleTransformModel.model_validate_json(model.model_dump_json())

        assert isinstance(restored.transform, TransformComposition)
        assert restored.transform.sequential_warp is True
        assert len(restored.transform.Ts) == 3
        assert isinstance(restored.transform.Ts[0], Translation)
        assert isinstance(restored.transform.Ts[1], SimilarityTransform)
        assert isinstance(restored.transform.Ts[2], InverseTransform)
        assert isinstance(restored.transform.Ts[2].T, _CustomNonInvertible)
        assert restored.transform.Ts[2].T.offset == 3.5

    def test_typed_field_omits_type_key(self) -> None:
        data = {
            "translation": {"t": (3.0, -7.0)},
            "similarity": {"s": 2.0, "r": 0.0, "t": (1.0, 2.0)},
            "affine": {"H": [[1.0, 0.0], [0.0, 1.0]], "t": (0.0, 0.0)},
        }
        model = _TypedTransformsModel.model_validate(data)
        assert model.translation == Translation((3.0, -7.0))
        assert model.similarity == SimilarityTransform(s=2.0, r=0.0, t=(1.0, 2.0))
        assert model.affine == AffineTransform(H=np.eye(2), t=(0.0, 0.0))

    def test_typed_field_rejects_wrong_subclass(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            _TypedTransformsModel.model_validate(
                {
                    # AffineTransform n'est pas une sous-classe de Translation
                    "translation": {"type": "AffineTransform", "H": [[1.0, 0.0], [0.0, 1.0]], "t": (0.0, 0.0)},
                    "similarity": {"s": 1.0, "r": 0.0, "t": (0.0, 0.0)},
                    "affine": {"H": [[1.0, 0.0], [0.0, 1.0]], "t": (0.0, 0.0)},
                }
            )

    def test_invalid_inputs_raise_validation_error(self) -> None:
        from pydantic import ValidationError

        # Clé "type" manquante sur un champ générique Transform
        with pytest.raises(ValidationError):
            _SingleTransformModel.model_validate({"transform": {"t": (1.0, 2.0)}})

        # Type inconnu dans le registre
        with pytest.raises(ValidationError):
            _SingleTransformModel.model_validate({"transform": {"type": "NonExistentTransform"}})

        # Type de donnée invalide (ni Transform, ni dict)
        with pytest.raises(ValidationError):
            _SingleTransformModel.model_validate({"transform": "not_a_transform"})
