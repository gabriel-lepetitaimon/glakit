from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, overload

import numpy as np
import numpy.typing as npt

from .geometry import (
    IntPoint,
    IntPointLike,
    IntRect,
    IntRectLike,
    Point,
    PointArray,
    PointArrayLike,
    PointArrayLikeStrict,
    PointLike,
    PointLikeStrict,
    Rect,
    RectLike,
)
from .safe_import import import_cv2
from .typing import Bool2DArray, Float1DArray, Float2DArray

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema


def _np_short_str(arr: npt.NDArray[np.floating]) -> str:
    if arr.ndim == 2:
        return "[" + "| ".join([" ".join(f"{v:.2f}" for v in row) for row in arr]) + "]"
    elif arr.ndim == 1:
        return "[" + " ".join(f"{v:.2f}" for v in arr) + "]"
    return str(arr)


class Transform(abc.ABC):
    # Registre automatique des sous-classes par nom
    _REGISTRY: ClassVar[dict[str, type[Transform]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Transform._REGISTRY[cls.__name__] = cls

    def to_dict(self) -> dict[str, Any]:
        """Sérialise la transformation en dictionnaire compatible JSON."""
        return {"type": self.__class__.__name__, **self._to_dict()}

    @abc.abstractmethod
    def _to_dict(self) -> dict[str, Any]:
        """Retourne les paramètres spécifiques de la sous-classe à sérialiser."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement '_to_dict_params'")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Reconstruit une instance à partir d'un dictionnaire."""
        data_dict = dict(data)
        type_name = data_dict.pop("type", None)

        if type_name is None:
            # Si on appelle directement Translation.from_dict({"t": (1, 2)}) sans préciser "type"
            if cls is Transform:
                raise ValueError("Missing 'type' key in Transform dictionary")
            target_cls = cls
        else:
            target_cls = Transform._REGISTRY.get(type_name)
            if target_cls is None:
                raise ValueError(f"Unknown Transform type: {type_name!r}")
            if not issubclass(target_cls, cls):
                raise TypeError(f"Expected {cls.__name__}, got {target_cls.__name__}")

        return target_cls._from_dict(data_dict)  # type: ignore

    @classmethod
    def _from_dict(cls, params: dict[str, Any]) -> Self:
        """Instancie la classe concrète à partir de ses paramètres désérialisés."""
        return cls(**params)

    @classmethod
    def identity(cls) -> Transform:
        """
        Returns the identity projection model.

        Returns
        -------
        projection : Transform
            The identity projection model.
        """
        return IdentityTransform()

    @classmethod
    def fit(cls, src: PointArrayLike, dst: PointArrayLike) -> tuple[Self, float]:
        """
        Fits a projection model to map points from ``src`` to ``dst``.

        Parameters
        ----------
        src : PointArrayLike
            The source points coordinates (N x 2) where N is the number of points.

        dst : PointArrayLike
            The destination points coordinates (N x 2).

        Returns
        -------
        projection : Self
            The fitted projection model.

        error : float
            The mean square error of the fitted model.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement the 'fit' method")

    @classmethod
    def fit_to_projection(
        cls,
        src: PointArrayLike,
        dst: PointArrayLike,
        projection: type[Self] | dict[int, type[Self]],
    ) -> tuple[Self, float]:
        """Fits a given projection model to map points from ``src`` to ``dst``.

        Parameters
        ----------
        src : PointArrayLike
            The source points coordinates (N, 2) where N is the number of points.

        dst : PointArrayLike
            The destination points coordinates (N, 2).

        projection : type[Self] | dict[int, type[Self]]
            The projection model to fit. If a dictionary is provided, the key is the minimum number of inliers required to use the corresponding projection.

        Returns
        -------
        tuple[Self, float]
            The fitted projection model and the mean square error of the fitted model.
        """
        src, dst = PointArray.as_array(src), PointArray.as_array(dst)
        if isinstance(projection, Mapping):
            projection = {k: projection[k] for k in sorted(projection.keys(), reverse=True)}

            n_inliers = len(src)
            proj = None
            for k, p in projection.items():
                proj = p
                if k <= n_inliers:
                    break
            else:
                raise ValueError("No projection model matches the number of inliers")
            return proj.fit(src, dst)
        elif issubclass(projection, Transform):
            return projection.fit(src, dst)

        raise ValueError("projection must be a projection model or a dictionary of projection models")

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Transform):
            return False
        return (self.invert() @ value).is_identity()

    def compose(self, T1: Transform) -> Transform:
        """
        Composes this projection model with another one.

        Parameters
        ----------
        T1 : Transform
            The other projection model to compose with.

        Returns
        -------
        T : Transform
            The composed projection model: T = self @ T1.
        """
        return TransformComposition.simplify_composition(self, T1)

    def _compose(self, T1: Transform) -> Transform | None:
        """Composes this projection model with another one and return a single Transform (not a TransformComposition) if they are composable or None otherwise.
        This method is called by the default implementation of the compose method and can be overridden by subclasses to implement specific composition rules."""
        return None

    def __matmul__(self, T1: Transform) -> Transform:
        """
        Composes this projection model with another one using the @ operator.

        Parameters
        ----------
        T1 : Transform
            The other projection model to compose with.

        Returns
        -------
        T : Transform
            The composed projection model: T = self @ T1.
        """
        return T1.compose(self)

    def invert(self) -> Transform:
        """
        Inverts this projection model.

        Returns
        -------
        T : Transform
            The inverted projection model: T = self^(-1)
        """
        return InverseTransform(self)

    def simplify(self) -> Transform:
        """Simplifies this projection model by trying to find a simpler projection model that is equivalent to this one."""
        return IdentityTransform() if self.is_identity() else self

    @property
    def is_exact(self) -> bool:
        """Whether this projection model is exact."""
        return True

    @property
    def is_inverse_exact(self) -> bool:
        """Whether the inverse of this projection model is exact."""
        return True

    def is_identity(self) -> bool:
        """Whether this projection model is the identity projection."""
        return False

    @property
    def is_yx_independent(self) -> bool:
        """
        Whether this projection model is independent on the y coordinate for the x coordinate and vice versa.
        (I.e. it can be decomposed into two 1D projections).
        """
        return False

    @overload
    def __call__(self, src: PointLikeStrict) -> Point: ...
    @overload
    def __call__(self, src: PointArrayLikeStrict) -> PointArray[Any]: ...
    def __call__(self, src: PointArrayLike) -> Point | PointArray[Any]:
        try:
            pt = Point.as_point(src, copy=False)
            return self.transform(pt)[0]
        except (ValueError, TypeError):
            return self.transform(src)

    @abc.abstractmethod
    def transform(self, src: PointArrayLike) -> PointArray[Any]:
        """
        Transforms a set of points with this projection model.

        Parameters
        ----------
        src : PointArrayLike
            The source points coordinates (N x 2) where N is the number of points.

        Returns
        -------
        dst : PointArray
            The transformed points coordinates (N x 2).
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement the 'transform' method")

    def transform_inverse(self, dst: PointArrayLike) -> PointArray[Any]:
        """
        Transforms a set of points with the inverse of this projection model.

        Parameters
        ----------
        dst : PointArrayLike
            The destination points coordinates (N x 2) where N is the number of points.

        Returns
        -------
        src : PointArray
            The source points coordinates (N x 2).
        """
        invert_t = self.invert()
        if isinstance(invert_t, InverseTransform):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not implement the 'transform_inverse' nor the 'invert' methods"
            )
        return invert_t.transform(dst)

    def transform_domain(self, moving_domain: RectLike) -> IntRect:
        """
        Transforms a domain with this projection model.

        Parameters
        ----------
        moving_domain : RectLike
            The source domain to transform.

        Returns
        -------
        IntRect
            The transformed domain.
        """
        domain = Rect.as_rect(moving_domain, copy=False)
        corners = self.transform(domain.corners())
        return IntRect.bounding_box(corners)

    def inverse_transform_domain(self, fixed_domain: RectLike) -> IntRect:
        """
        Transforms a domain with the inverse of this projection model.

        Parameters
        ----------
        fixed_domain : RectLike
            The destination domain to transform.

        Returns
        -------
        IntRect
            The transformed domain.
        """
        domain = Rect.as_rect(fixed_domain, copy=False)
        corners = self.transform_inverse(domain.corners())
        return IntRect.bounding_box(corners)

    @overload
    def quadratic_error(
        self, src: PointArrayLike, dst: PointArrayLike, mean: Literal[False] = False
    ) -> Float1DArray: ...
    @overload
    def quadratic_error(self, src: PointArrayLike, dst: PointArrayLike, mean: Literal[True]) -> float: ...
    def quadratic_error(self, src: PointArrayLike, dst: PointArrayLike, mean: bool = False) -> Float1DArray | float:
        """
        Calculates the quadratic error of the projection model when mapping points from ``src`` to ``dst``.

        Parameters
        ----------
        src : PointArrayLike
            The source points coordinates (N x 2) where N is the number of points.

        dst : PointArrayLike
            The destination points coordinates (N x 2).

        mean : bool, optional
            Whether to return the mean error. By default False.

        Returns
        -------
        error : Float1DArray | float
            The quadratic error of each point or the mean error if ``mean`` is True.
        """
        errors = self._quadratic_error(src, dst)
        return float(np.mean(errors)) if mean else errors

    def _quadratic_error(self, src: PointArrayLike, dst: PointArrayLike) -> Float1DArray:
        src, dst = PointArray.as_array(src), PointArray.as_array(dst)
        diff = (dst - self.transform(src)).view(np.ndarray)
        return np.sum(diff**2, axis=1)

    def warp[DTYPE: np.uint8 | np.float32 | np.bool_](
        self,
        src_img: npt.NDArray[DTYPE],
        src_top_left: IntPointLike = (0, 0),
        warped_domain: IntRectLike | Literal["full", "same"] = "full",
        channel_last: bool = True,
    ) -> tuple[npt.NDArray[DTYPE], IntRect]:
        """
        Warps an image using this projection model.

        Parameters
        ----------
        src_img : npt.NDArray[DTYPE]
            The source image to warp. Shape=(H x W [x C]) or shape=(C x H x W) if ``channel_last`` is False.

        src_top_left : IntPointLike
            The top-left corner of the source image domain. The ``src_domain`` is defined as a Rect with this top-left corner and the size of the source image.

        warped_domain : IntRectLike | Literal["full", "same"], optional
            The domain of the destination image.
            - "full": the destination domain is computed by transforming ``src_domain``;
            - "same": the destination domain is the same as ``src_domain``;
            - or any Rect manually defining the requested destination domain.

        channel_last : bool, optional
            Whether the source image has the channel dimension as the last axis. By default True.

        Returns
        -------
        dst_img : npt.NDArray[DTYPE]
            The warped image.

        warped_domain : IntRect
            The domain of the warped image.
        """
        src_origin = -IntPoint.as_point(src_top_left)
        if not channel_last and src_img.ndim == 3:
            src_img = np.transpose(src_img, (1, 2, 0))

        warped_domain, src_region, src_region_domain = self.select_warped_region(src_img, src_origin, warped_domain)
        is_bool = src_region.dtype == bool
        if is_bool:
            src_region = (src_region * np.uint8(255)).astype(np.uint8)
        dst_map, warped_domain = self._warp(src_region, src_region_domain, warped_domain)  # type: ignore
        if is_bool:
            dst_map = dst_map > 125
        if not channel_last and dst_map.ndim == 3:
            dst_map = np.transpose(dst_map, (2, 0, 1))
        return dst_map, warped_domain  # type: ignore

    def _warp[DTYPE: np.uint8 | np.float32](
        self, src_img: npt.NDArray[DTYPE], src_domain: IntRect, dst_domain: IntRect
    ) -> tuple[npt.NDArray[DTYPE], IntRect]:
        cv2 = import_cv2()

        dst_grid = dst_domain.grid_indices(dtype=np.float32).reshape(-1, 2)
        src_remap = (self.transform_inverse(dst_grid) - src_domain.top_left).view(np.ndarray)
        src_remap = src_remap.reshape(*dst_domain.to_shape(), 2).astype(np.float32)
        return cv2.remap(src_img, src_remap[..., ::-1], None, cv2.INTER_LINEAR), dst_domain  # type: ignore

    def warped_domain(
        self,
        src_img_shape: IntPointLike,
        src_top_left: IntPointLike = (0, 0),
        warped_domain: IntRectLike | Literal["full", "same"] = "full",
    ) -> IntRect:
        """
        Computes the domain of the warped image using this projection model.

        Parameters
        ----------
        src_img_shape : Point | tuple[int, int]
            The shape of the source image to warp. The ``src_domain`` is defined as a Rect with the top-left corner defined by ``src_top_left`` and the size of the source image.

        src_top_left : Point | tuple[int, int]
            The top-left corner of the source image domain. The ``src_domain`` is defined as a Rect with this top-left corner and the size of the source image.

        warped_domain : Rect | Literal["full", "same"], optional
            The domain of the destination image.
            - "full": the destination domain is computed by transforming ``src_domain``;
            - "same": the destination domain is the same as ``src_domain``;
            - or any Rect manually defining the requested destination domain.

        Returns
        -------
        warped_domain : Rect
            The domain of the warped image.
        """
        src_domain = IntRect(src_img_shape) + IntPoint.as_point(src_top_left)
        if warped_domain == "full":
            return self.transform_domain(src_domain)
        elif warped_domain == "same":
            return src_domain
        return IntRect.as_rect(warped_domain)

    def select_warped_region[T: np.generic](
        self,
        src_img: npt.NDArray[T],
        src_origin: IntPointLike,
        warped_domain: IntRectLike | Literal["full", "same"],
    ) -> tuple[IntRect, npt.NDArray[T], IntRect]:
        """
        Selects a region to warp from an image using this projection model.

        Parameters
        ----------
        src_img : npt.NDArray[T]
            The source image to select the region from of shape (H, W[, C]).

        src_origin : Point | tuple[int, int]
            The origin of the coordinates system in the source image. The ``src_domain`` is defined as a Rect with this origin and the size of the source image.

        warped_domain : Rect | Literal["full", "same"]
            The domain of the region to select.

        Returns
        -------
        warped_domain : Rect
            The requested domain after the warp.

        src_region : npt.NDArray[T]
            The selected region from the source image to warp.

        src_region_domain : Rect
            The domain of the selected region in the source image. This domain is the intersection of the warped domain transformed back to the source image and the source image domain.
        """
        src_origin = IntPoint.as_point(src_origin)
        src_domain = IntRect((src_img.shape[0], src_img.shape[1])) - src_origin
        if warped_domain == "full":
            target_domain = self.transform_domain(src_domain)
        elif warped_domain == "same":
            target_domain = src_domain
        else:
            target_domain = IntRect.as_rect(warped_domain)

        src_region_domain = self.inverse_transform_domain(target_domain) & src_domain
        src_region = src_region_domain.crop_pad_image(src_img, origin=src_origin, channel_last=True, copy=False)
        return target_domain, src_region, src_region_domain

    def draw_grid(
        self,
        domain: RectLike,
        subdivision: int | tuple[int, int] | None = None,
        subsampling: int = 1,
        inverse: bool = False,
    ) -> tuple[Bool2DArray, IntRect]:
        """
        Draws a grid on a domain using this projection model.

        Parameters
        ----------
        domain : Rect
            The domain to draw the grid on.

        subdivision : int | tuple[int, int]
            The number of subdivisions of the grid. If an integer is provided, it is used for both dimensions. If None, set the subdivision to approximately have a grid line every 100 pixels in the original domain.

        subsampling : int
            The subsampling factor to apply to the grid. The grid is drawn on a subsampled version of the domain and then upsampled back to the original resolution. This can be used to speed up the drawing of the grid.

        inverse : bool
            Whether to draw the grid using the inverse of this projection model. By default False.

        Returns
        -------
        grid : Bool2DArray
            A boolean array where True values correspond to the grid lines.

        grid_domain : Rect
            The domain of the grid.
        """
        domain = Rect.as_rect(domain)
        if isinstance(subdivision, int):
            subdivision = (subdivision, subdivision)
        elif subdivision is None:
            subdivision = (int(max(1, domain.w // 100)), int(max(1, domain.h // 100)))
        assert isinstance(subdivision, tuple) and len(subdivision) == 2, (
            "subdivision must be an int or a tuple of two ints"
        )
        assert subdivision[0] > 0 and subdivision[1] > 0, "subdivision must be positive"

        sub_y = np.linspace(domain.top, domain.bottom, subdivision[0] + 2)
        h_line = np.arange(subsampling, sub_y[1] - sub_y[0], subsampling)
        h_line = np.concatenate([sub_y] + [h_line + sub_y[i] for i in range(len(sub_y))])

        sub_x = np.linspace(domain.left, domain.right, subdivision[1] + 2)
        v_line = np.arange(subsampling, sub_x[1] - sub_x[0], subsampling)
        v_line = np.concatenate([sub_x] + [v_line + sub_x[i] for i in range(len(sub_x))])

        grid_pts = []
        for y in sub_y:
            grid_pts.append(np.column_stack((np.full_like(v_line, y), v_line)))
        for x in sub_x:
            grid_pts.append(np.column_stack((h_line, np.full_like(h_line, x))))
        grid_pts = PointArray(np.concatenate(grid_pts, axis=0), copy=False)

        if not inverse:
            grid_pts = self.transform(grid_pts)
        else:
            grid_pts = self.transform_inverse(grid_pts)

        grid_domain = IntRect.bounding_box(grid_pts)
        grid_yx = np.round((grid_pts - grid_domain.top_left).view(np.ndarray)).astype(int)

        grid: Bool2DArray = np.zeros(grid_domain.pad(0, 0, 1, 1).to_shape(), dtype=bool)
        grid[*grid_yx.T] = True
        return grid, grid_domain

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        from pydantic_core import core_schema

        def validate(value: Any) -> Self:
            if isinstance(value, cls):
                return value
            if isinstance(value, Mapping):
                try:
                    return cls.from_dict(value)
                except (KeyError, TypeError, ValueError) as e:
                    raise ValueError(str(e)) from e
            raise ValueError(f"Cannot validate {type(value).__name__} as {cls.__name__}")

        # Schéma JSON : un objet dictionnaire contenant au minimum les clés validées par `validate`
        json_schema = core_schema.no_info_after_validator_function(
            validate,
            core_schema.dict_schema(keys_schema=core_schema.str_schema(), values_schema=core_schema.any_schema()),
        )

        return core_schema.json_or_python_schema(
            json_schema=json_schema,
            python_schema=core_schema.no_info_plain_validator_function(validate),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda t: t.to_dict(),
                return_schema=core_schema.dict_schema(),
            ),
        )


class TransformComposition(Transform):
    def __init__(self, *Ts: Transform, sequential_warp: bool = False) -> None:
        self.Ts = Ts
        self.sequential_warp = sequential_warp
        super().__init__()

    def _to_dict(self) -> dict[str, Any]:
        return {
            "Ts": [t.to_dict() for t in self.Ts],
            "sequential_warp": self.sequential_warp,
        }

    @classmethod
    def _from_dict(cls, params: dict[str, Any]) -> Self:
        Ts = [Transform.from_dict(t) for t in params["Ts"]]
        return cls(*Ts, sequential_warp=params.get("sequential_warp", False))

    def __repr__(self) -> str:
        return f"ProjectionComposition({', '.join(repr(T) for T in self.Ts)})"

    def __str__(self) -> str:
        return " @ ".join(str(T) for T in reversed(self.Ts))

    @staticmethod
    def simplify_composition(*Ts: Transform) -> Transform:
        expanded_transforms: list[Transform] = []
        for T in Ts:
            if isinstance(T, TransformComposition):
                expanded_transforms.extend(T.Ts)
            elif not isinstance((T := T.simplify()), IdentityTransform):
                expanded_transforms.append(T)
        Ts_ = list(expanded_transforms)

        simplified = True
        while simplified:
            simplified = False
            i = 0
            while i < len(Ts_) - 1:
                T1, T2 = Ts_[i], Ts_[i + 1]
                if T1.is_identity():
                    simplified = True
                    del Ts_[i]
                elif (T1T2 := T1._compose(T2)) is not None:
                    simplified = True
                    Ts_[i] = T1T2
                    del Ts_[i + 1]
                elif (isinstance(T1, InverseTransform) and T1.T is T2) or (
                    isinstance(T2, InverseTransform) and T2.T is T1
                ):
                    simplified = True
                    del Ts_[i + 1]
                    del Ts_[i]
                else:
                    i += 1

        if not Ts_:
            return IdentityTransform()
        if len(Ts_) == 1:
            return Ts_[0]
        return TransformComposition(*Ts_)

    def compose(self, T1: Transform) -> Transform:
        if isinstance(T1, TransformComposition):
            return TransformComposition.simplify_composition(*self.Ts, *T1.Ts)
        return TransformComposition.simplify_composition(*self.Ts, T1)

    def simplify(self) -> Transform:
        return TransformComposition.simplify_composition(*self.Ts)

    @property
    def is_exact(self) -> bool:
        return all(T.is_exact for T in self.Ts)

    @property
    def is_inverse_exact(self) -> bool:
        return all(T.is_inverse_exact for T in self.Ts)

    @property
    def is_yx_independent(self) -> bool:
        return all(T.is_yx_independent for T in self.Ts)

    def invert(self) -> Self:
        return type(self)(*(T.invert() for T in reversed(self.Ts)))

    def transform(self, src: PointArrayLike) -> PointArray[Any]:
        pts = PointArray.as_array(src)
        for T in self.Ts:
            pts = T.transform(pts)
        return pts

    def transform_inverse(self, dst: PointArrayLike) -> PointArray[Any]:
        pts = PointArray.as_array(dst)
        for T in reversed(self.Ts):
            pts = T.transform_inverse(pts)
        return pts

    def _warp[DTYPE: np.uint8 | np.float32](
        self, src_img: npt.NDArray[DTYPE], src_domain: IntRect, dst_domain: IntRect
    ) -> tuple[npt.NDArray[DTYPE], IntRect]:
        if not self.sequential_warp:
            return super()._warp(src_img, src_domain, dst_domain)

        for T in self.Ts:
            src_img, src_domain = T._warp(src_img, src_domain, dst_domain)
        return src_img, src_domain


class InverseTransform(Transform):
    def __init__(self, T: Transform) -> None:
        self.T = T
        super().__init__()

    def _to_dict(self) -> dict[str, Any]:
        return {"T": self.T.to_dict()}

    @classmethod
    def _from_dict(cls, params: dict[str, Any]) -> Self:
        return cls(T=Transform.from_dict(params["T"]))

    def __repr__(self) -> str:
        return f"ProjectionInverse({self.T})"

    def __str__(self) -> str:
        return f"Inv[{self.T}]"

    def invert(self) -> Transform:
        return self.T

    def simplify(self) -> Transform:
        if isinstance(self.T, InverseTransform):
            return self.T.T.simplify()
        if not isinstance(invT := self.T.invert(), InverseTransform):
            return invT.simplify()
        return self

    def is_identity(self) -> bool:
        return self.T.is_identity()

    @property
    def is_exact(self) -> bool:
        return self.T.is_inverse_exact

    @property
    def is_inverse_exact(self) -> bool:
        return self.T.is_exact

    @property
    def is_yx_independent(self) -> bool:
        return self.T.is_yx_independent

    def transform(self, src: PointArrayLike) -> PointArray[Any]:
        return self.T.transform_inverse(src)

    def transform_inverse(self, dst: PointArrayLike) -> PointArray[Any]:
        return self.T.transform(dst)


class IdentityTransform(Transform):
    def __init__(self) -> None:
        super().__init__()

    def _to_dict(self) -> dict[str, Any]:
        return {}

    def __repr__(self) -> str:
        return "IdentityProjection()"

    def __str__(self) -> str:
        return "I"

    def is_identity(self) -> bool:
        return True

    def simplify(self) -> Transform:
        return self

    @property
    def is_exact(self) -> bool:
        return True

    @property
    def is_inverse_exact(self) -> bool:
        return True

    @property
    def is_yx_independent(self) -> bool:
        return True

    def invert(self) -> Self:
        return self

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Transform):
            return False
        return value.is_identity()

    def _compose(self, T1: Transform) -> Transform | None:
        return T1

    def transform(self, src: PointArrayLike) -> PointArray[Any]:
        return PointArray.as_array(src, copy=True)

    def transform_inverse(self, dst: PointArrayLike) -> PointArray[Any]:
        return PointArray.as_array(dst, copy=True)

    def _warp[DTYPE: np.uint8 | np.float32](
        self, src_img: npt.NDArray[DTYPE], src_domain: IntRect, dst_domain: IntRect
    ) -> tuple[npt.NDArray[DTYPE], IntRect]:
        dst_img = dst_domain.crop_pad_image(src_img, origin=-src_domain.top_left, channel_last=True, copy=False)
        return dst_img, dst_domain


class AffineTransform(Transform):
    H: Float2DArray
    t: Point

    def __init__(self, H: npt.NDArray[np.floating], t: PointLike) -> None:
        assert H.shape == (2, 2), "H must be a 2x2 matrix"
        self.H = H.astype(np.float64)
        self.t = Point.as_point(t)
        super().__init__()

    def _to_dict(self) -> dict[str, Any]:
        return {"H": self.H.tolist(), "t": self.t.tuple}

    @classmethod
    def _from_dict(cls, params: dict[str, Any]) -> Self:
        return cls(H=np.asarray(params["H"], dtype=np.float64), t=params["t"])

    def __repr__(self) -> str:
        return f"AffineProjection(H={self.H}, t={self.t})"

    def __str__(self) -> str:
        return f"Affine(H={_np_short_str(self.H)}, t={_np_short_str(self.t.view(np.ndarray))})"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, AffineTransform):
            return False
        return np.allclose(self.H, value.H) and self.t == value.t

    @staticmethod
    def rotate(theta: float, center: PointLike = (0, 0)) -> AffineTransform:
        """Create an affine transformation that rotates by theta and translates by t.

        Parameters
        ----------
        theta : float
            Rotation angle in degrees. Positive values rotate clockwise.
        center : PointLike
            Center of rotation.

        Returns
        -------
        AffineTransform
            The corresponding affine transformation.
        """
        R = AffineTransform.rotation_matrix(-theta, deg=True)
        center_pt = Point.as_point(center)
        t = center_pt - R @ center_pt
        return AffineTransform(R, t)

    @staticmethod
    def rotation_matrix(theta: float, deg: bool = False) -> Float2DArray:
        """Create a rotation matrix for a given angle."""
        if deg:
            theta = np.deg2rad(theta)
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])  # type: ignore

    @classmethod
    def fit(cls, src: PointArrayLike, dst: PointArrayLike) -> tuple[Self, float]:
        src = PointArray.as_array(src).view(np.ndarray)
        dst = PointArray.as_array(dst).view(np.ndarray)
        assert src.shape == dst.shape, "src and dst must have the same shape"
        src_hom = np.concatenate((src, np.ones((src.shape[0], 1))), axis=1)
        X, _, _, _ = np.linalg.lstsq(src_hom, dst, rcond=None)  # type: ignore
        R, t = X[:2].T, X[2]
        error = np.sum((dst - src_hom @ X) ** 2, axis=1)
        return cls(R, t), float(np.mean(error))

    if TYPE_CHECKING:

        @overload
        def compose(self, T1: AffineTransform) -> AffineTransform: ...
        @overload
        def compose(self, T1: Transform) -> Transform: ...
        def compose(self, T1: Transform) -> Transform: ...

        @overload
        def __matmul__(self, T1: AffineTransform) -> AffineTransform: ...
        @overload
        def __matmul__(self, T1: Transform) -> Transform: ...
        def __matmul__(self, T1: Transform) -> Transform: ...

    def _compose(self, T1: Transform) -> Transform | None:
        if isinstance(T1, AffineTransform):
            return AffineTransform(T1.H @ self.H, T1.H @ self.t + T1.t)
        return None

    def is_identity(self) -> bool:
        return bool(np.allclose(self.H, np.eye(2)) and self.t == Point.origin())

    @property
    def is_yx_independent(self) -> bool:
        return bool(np.allclose(self.H[[0, 1], [1, 0]], [0, 0]))

    def invert(self) -> AffineTransform:
        """Invert the affine transformation."""
        cv2 = import_cv2()
        M = cv2.invertAffineTransform(self.M)
        return AffineTransform(H=M[:2, :2], t=M[:2, 2])  # type: ignore

    def transform(self, src: PointArrayLike) -> PointArray[Any]:
        src_arr = PointArray.as_array(src).view(np.ndarray)
        return PointArray(src_arr @ self.H.T, copy=False) + self.t

    @property
    def M(self) -> Float2DArray:
        return np.concatenate((self.H, self.t.view(np.ndarray)[:, None]), axis=1)  # type: ignore

    @property
    def scaling(self) -> float:
        """The overall scaling factor of this affine transformation."""
        det = float(np.linalg.det(self.H))
        scaling = float(np.sqrt(np.abs(det)))
        return scaling if det >= 0 else -scaling

    def as_translation(self) -> Translation | None:
        """If this affine transformation is a translation, returns the corresponding Translation object."""
        if not np.allclose(self.H, np.eye(2)):
            return None
        return Translation(t=self.t)

    def as_similarity_transform(self) -> SimilarityTransform | None:
        """If this affine transformation is a similarity transformation, returns the corresponding SimilarityTransform object."""
        if not np.allclose(self.H[0, 0], self.H[1, 1]) or not np.allclose(self.H[0, 1], -self.H[1, 0]):
            return None
        s = float(np.sqrt(self.H[0, 0] ** 2 + self.H[1, 0] ** 2))
        r = float(np.arctan2(self.H[1, 0] / s, self.H[0, 0] / s))
        return SimilarityTransform(s=s, r=r, t=self.t)

    def as_resize_translation(self) -> ResizeTranslation | None:
        """If this affine transformation is a resize and translation transformation, returns the corresponding ResizeTranslation object."""
        if not np.allclose(self.H[[0, 1], [1, 0]], 0):
            return None
        return ResizeTranslation(s=self.H.diagonal(), t=self.t)

    def _warp[DTYPE: np.uint8 | np.float32](
        self, src_img: npt.NDArray[DTYPE], src_domain: IntRect, dst_domain: IntRect
    ) -> tuple[npt.NDArray[DTYPE], IntRect]:
        cv2 = import_cv2()

        T = Translation(src_domain.top_left) @ self @ Translation(-dst_domain.top_left)
        M_xy = T.M[[1, 0]][:, [1, 0, 2]]  # OpenCV uses (x, y) coordinates while we use (y, x) coordinates
        return cv2.warpAffine(src_img, M_xy, dst_domain.sizes.xy, flags=cv2.INTER_LINEAR), dst_domain  # type: ignore


class FlipTransform(AffineTransform):
    center: Point
    horizontal: bool
    vertical: bool

    def __init__(self, center: PointLike, horizontal: bool = True, vertical: bool = False) -> None:
        """
        A projection model that flips points horizontally and/or vertically around a center point.

        Parameters
        ----------
        center : PointLike
            The center point (y, x) around which to flip the points.
        horizontal : bool, optional
            Whether to flip points horizontally. Default is True.
        vertical : bool, optional
            Whether to flip points vertically. Default is False.
        """
        self.horizontal = horizontal
        self.vertical = vertical
        self.center = Point.as_point(center)
        super().__init__(
            H=np.diag([1 - 2 * vertical, 1 - 2 * horizontal]),
            t=Point(2 * vertical * self.center.y, 2 * horizontal * self.center.x),
        )

    def _to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tuple,
            "horizontal": self.horizontal,
            "vertical": self.vertical,
        }

    def __repr__(self) -> str:
        return f"FlipProjection(center={self.center}, horizontal={self.horizontal}, vertical={self.vertical})"

    def __str__(self) -> str:
        flips = []
        if self.horizontal:
            flips.append("H")
        if self.vertical:
            flips.append("V")
        return "Flip(" + ",".join(flips) + ")"

    def is_identity(self) -> bool:
        return not self.horizontal and not self.vertical

    @property
    def is_yx_independent(self) -> bool:
        return True

    def invert(self) -> Self:
        return self.__class__(self.center, self.horizontal, self.vertical)

    def transform(self, src: PointArrayLike) -> PointArray[Any]:
        dst = PointArray.as_array(src, copy=True)
        if self.horizontal:
            dst.x[:] = 2 * self.center.x - dst.x
        if self.vertical:
            dst.y[:] = 2 * self.center.y - dst.y
        return dst

    def transform_inverse(self, dst: PointArrayLike) -> PointArray[Any]:
        return self.transform(dst)

    def _warp[DTYPE: np.uint8 | np.float32](
        self, src_img: npt.NDArray[DTYPE], src_domain: IntRect, dst_domain: IntRect
    ) -> tuple[npt.NDArray[DTYPE], IntRect]:
        dst_domain = self.transform_domain(src_domain)

        dst_img = src_img
        if self.horizontal:
            dst_img = np.fliplr(dst_img)
        if self.vertical:
            dst_img = np.flipud(dst_img)

        dst = dst_domain.crop_pad_image(dst_img, origin=dst_domain.top_left, channel_last=True, copy=False)
        return dst, dst_domain  # type: ignore


class SimilarityTransform(AffineTransform):
    s: float
    """The scaling factor of the similarity transform."""
    r: float
    """The rotation angle of the similarity transform in radians."""
    t: Point
    """The translation vector of the similarity transform."""

    def __init__(self, s: float = 1, r: float = 0, t: PointLike | None = None) -> None:
        """A projection model that applies a similarity transform to points.

        Parameters
        ----------
        s : float, optional
            The scaling factor. Default is 1 (no scaling).
        r : float, optional
            The rotation angle in radians. Default is 0 (no rotation).
        t : PointLike, optional
            The translation vector. Default is None (no translation).
        """
        assert s > 0, "s must be positive"
        self.t = Point.as_point(t) if t is not None else Point.origin()
        self.r = r
        self.s = s
        super().__init__(SimilarityTransform.rotation_scale_matrix(r, s), self.t)

    def _to_dict(self) -> dict[str, Any]:
        return {"s": self.s, "r": self.r, "t": self.t.tuple}

    @staticmethod
    def rotation_scale_matrix(theta: float, scale: float, deg: bool = False) -> Float2DArray:
        return scale * AffineTransform.rotation_matrix(theta, deg=deg)  # type: ignore

    @classmethod
    def fit(cls, src: PointArrayLike, dst: PointArrayLike) -> tuple[Self, float]:
        src, dst = PointArray.as_array(src).view(np.ndarray), PointArray.as_array(dst).view(np.ndarray)
        assert src.shape == dst.shape, "src and dst must have the same shape"
        assert len(src) >= 2, "At least 2 points are required to fit a similarity transform"

        src_barycenter = Point.as_point(np.mean(src, axis=0))
        dst_barycenter = Point.as_point(np.mean(dst, axis=0))
        src_, dst_ = src - src_barycenter.view(np.ndarray), dst - dst_barycenter.view(np.ndarray)

        src_norm = np.linalg.norm(src_)
        dst_norm = np.linalg.norm(dst_)
        s = float(dst_norm / src_norm) if src_norm > 0 else 1.0
        r = float(
            np.arctan2(
                np.sum(src_[:, 0] * dst_[:, 1] - src_[:, 1] * dst_[:, 0]),
                np.sum(src_ * dst_),
            )
        )

        R = SimilarityTransform.rotation_scale_matrix(r, s)
        t = dst_barycenter - (src_barycenter @ R.T)
        mse = float(np.sum((dst - src @ R.T - t.view(np.ndarray)) ** 2, axis=1).mean())
        return cls(s, r, t), mse

    def __repr__(self) -> str:
        return f"SimilarityTransform(s={self.s}, r={self.r}, t={self.t})"

    def __str__(self) -> str:
        return f"Sim(s={self.s:.2f}, r={self.r:.2f}, t={_np_short_str(self.t.view(np.ndarray))})"

    def is_identity(self) -> bool:
        return self.r == 0 and self.s == 1 and self.t == Point.origin()

    @property
    def is_yx_independent(self) -> bool:
        return self.r == 0

    def invert(self) -> SimilarityTransform:
        """
        Returns the inverse of this similarity transform.

        Example
        -------
        >>> T = SimilarityTransform(s=2, r=np.pi/4, t=(1, 1))
        >>> T_inv = T.invert()
        >>> src = np.array([[0, 0], [1, 0], [0, 1]])
        >>> dst = T.transform(src)
        >>> np.allclose(T_inv.transform(dst), src)
        True
        """
        return SimilarityTransform(s=1 / self.s, r=-self.r, t=self.transform_inverse(np.zeros(2))[0])

    def _compose(self, T1: Transform) -> Transform | None:
        """Composes this similarity transform with another projection model.
        If the other projection model is also a similarity transform, the composition is simplified to a single similarity transform. Otherwise, the composition is returned as a ProjectionComposition.

        Parameters
        ----------
        T1 : FundusProjection
            The other projection model to compose with.

        Returns
        -------
        FundusProjection
            The composed projection model: T = self @ T1

        Example
        -------
        >>> T1 = SimilarityTransform(s=2, r=np.pi/4, t=(1, 1))
        >>> T1.compose(T1.invert()).is_identity()
        True
        """
        if isinstance(T1, SimilarityTransform):
            return SimilarityTransform(s=self.s * T1.s, r=self.r + T1.r, t=T1(self.t))
        return super()._compose(T1)

    def transform_inverse(self, dst: PointArrayLike) -> PointArray[Any]:
        """
        Transforms a set of points with the inverse of this projection model.

        Parameters
        ----------
        dst : npt.NDArray[np.floating]
            The destination points coordinates (N x 2) where N is the number of points.

        Returns
        -------
        src : npt.NDArray[np.floating]
            The source points coordinates (N x 2).

        Example
        -------
        >>> T = SimilarityTransform(s=2, r=np.pi/4, t=(1, 1))
        >>> src = np.array([[0, 0], [1, 0], [0, 1]])
        >>> dst = T.transform(src)
        >>> np.allclose(T.transform_inverse(dst), src)
        True
        """
        dst_pts = PointArray.as_array(dst)
        R = SimilarityTransform.rotation_scale_matrix(-self.r, 1 / self.s)
        return PointArray((dst_pts - self.t).view(np.ndarray) @ R.T, copy=False)


class ResizeTranslation(AffineTransform):
    t: Point
    s: Point

    def __init__(self, s: float | PointLike, t: PointLike | None = None) -> None:
        """A projection model that applies a resize and a translation to points.
        Parameters
        ----------
        s : float | PointLike
            The scaling factor. Must be positive.

        t : PointLike, optional
            The translation vector. If None, no translation is applied.
        """
        s = Point.as_point(s)
        assert s.y > 0 and s.x > 0, "s must be positive"

        self.s = s
        self.t = Point.as_point(t) if t is not None else Point.origin()
        super().__init__(self.scale_matrix(self.s), self.t)

    def _to_dict(self) -> dict[str, Any]:
        return {"s": self.s.tuple, "t": self.t.tuple}

    @classmethod
    def scale_matrix(cls, s: float | PointLike) -> Float2DArray:
        """Create a scaling matrix for a given scale factor.

        Parameters
        ----------
        s : float | FloatPairLike
            The scaling factor. Must be positive. If a single float is provided, it is used for both dimensions.

        Returns
        -------
        Float2DArray
            The corresponding scaling matrix.
        """
        return np.diag(Point.as_point(s).view(np.ndarray))  # type: ignore

    @classmethod
    def translate_resize(cls, t: PointLike, s: float | PointLike) -> Self:
        s = Point.as_point(s)
        return cls(s, Point.as_point(t) * s)

    @classmethod
    def resize(cls, s: float | PointLike, center: PointLike = (0, 0)) -> Self:
        s = Point.as_point(s)
        return cls(s, Point.as_point(center) * (1 - s))

    @classmethod
    def fit(cls, src: PointArrayLike, dst: PointArrayLike, *, uniform_scale: bool = True) -> tuple[Self, float]:
        """Fits a resize and translation projection model to a set of corresponding points.

        Parameters
        ----------
        src : PointArrayLike
            The source points coordinates (N x 2) where N is the number of points.
        dst : PointArrayLike
            The destination points coordinates (N x 2) where N is the number of points.
        uniform_scale : bool, optional
            Whether to use a uniform scale factor for both dimensions. If False, a separate scale factor is computed for each dimension. Default is True.

        Returns
        -------
        T : ResizeTranslation
            The fitted resize and translation projection model.
        error : float
            The mean squared error of the fitted model on the provided points.
        """
        src_arr = PointArray.as_array(src).ndarray
        dst_arr = PointArray.as_array(dst).ndarray
        assert src_arr.shape == dst_arr.shape, "src and dst must have the same shape"

        if uniform_scale:
            N = src_arr.shape[0]
            A = np.zeros((2 * N, 3))
            b = np.zeros((2 * N,))
            A[0::2, 0] = src_arr[:, 0]
            A[0::2, 1] = 1
            A[1::2, 0] = src_arr[:, 1]
            A[1::2, 2] = 1
            b[0::2] = dst_arr[:, 0]
            b[1::2] = dst_arr[:, 1]
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            s = Point(x[0], x[0])
            t = Point(x[1], x[2])
        else:
            src_mean = np.mean(src_arr, axis=0)
            dst_mean = np.mean(dst_arr, axis=0)
            src_centered = src_arr - src_mean
            dst_centered = dst_arr - dst_mean
            denom = np.sum(src_centered**2, axis=0)
            s_vec = np.where(denom > 0, np.sum(src_centered * dst_centered, axis=0) / denom, 1.0)
            t_vec = dst_mean - s_vec * src_mean
            s = Point.as_point(s_vec)
            t = Point.as_point(t_vec)

        error = np.sum((dst_arr - (s.view(np.ndarray) * src_arr + t.view(np.ndarray))) ** 2, axis=1)
        return cls(s, t), float(np.mean(error))

    def __repr__(self) -> str:
        return f"ResizeTranslateProjection(s={self.s}, t={self.t})"

    def __str__(self) -> str:
        s = f"{self.s.y}" if self.s.y == self.s.x else f"s={_np_short_str(self.s.view(np.ndarray))}"
        return f"ResizeTranslateProjection(s={s}, t={_np_short_str(self.t.view(np.ndarray))})"

    def invert(self) -> Self:
        return self.__class__(1 / self.s, -self.t / self.s)

    def is_identity(self) -> bool:
        return self.s == Point(1.0, 1.0) and self.t == Point.origin()

    @property
    def is_yx_independent(self) -> bool:
        return True

    def _compose(self, T1: Transform) -> Transform | None:
        if isinstance(T1, ResizeTranslation):
            return ResizeTranslation(s=self.s * T1.s, t=T1(self.t))
        return super()._compose(T1)

    def transform(self, src: PointArrayLike) -> PointArray[Any]:
        return self.s * PointArray.as_array(src) + self.t

    def transform_inverse(self, dst: PointArrayLike) -> PointArray[Any]:
        return (PointArray.as_array(dst) - self.t) / self.s

    def _warp[DTYPE: np.uint8 | np.float32](
        self, src_img: npt.NDArray[DTYPE], src_domain: IntRect, dst_domain: IntRect
    ) -> tuple[npt.NDArray[DTYPE], IntRect]:
        from .safe_import import cv2

        dst_region_domain = self.transform_domain(src_domain)
        if np.issubdtype(src_img.dtype, np.integer):
            interpolation = cv2.INTER_NEAREST
        else:
            interpolation = cv2.INTER_AREA if abs(self.s.y) < 1 else cv2.INTER_LINEAR
        dst_region = cv2.resize(src_img, dst_domain.sizes.xy, fx=self.s.x, fy=self.s.y, interpolation=interpolation)
        dst = dst_domain.crop_pad_image(dst_region, origin=-dst_region_domain.top_left, channel_last=True, copy=False)
        return dst, dst_domain  # type: ignore


class Translation(ResizeTranslation):
    """A projection model that translates points by a given vector."""

    def __init__(self, t: PointLike) -> None:
        """Initializes a Translation projection model.

        Parameters
        ----------
        t : FloatPairLike
            The translation vector.
        """
        super().__init__(t=t, s=1)

    def _to_dict(self) -> dict[str, Any]:
        return {"t": self.t.tuple}

    def __repr__(self) -> str:
        return f"Translation(t={self.t})"

    def __str__(self) -> str:
        return f"Trans(t={_np_short_str(self.t.view(np.ndarray))})"

    def is_identity(self) -> bool:
        return self.t == Point.origin()

    def invert(self) -> Self:
        return self.__class__(-self.t)

    def transform(self, src: PointArrayLike) -> PointArray[Any]:
        return PointArray.as_array(src) + self.t

    def transform_inverse(self, dst: PointArrayLike) -> PointArray[Any]:
        return PointArray.as_array(dst) - self.t

    def _compose(self, T1: Transform) -> Transform | None:
        if isinstance(T1, Translation):
            return Translation(self.t + T1.t)
        return super()._compose(T1)

    @classmethod
    def identity(cls) -> Transform:
        return cls(Point.origin())

    @classmethod
    def fit(cls, src: PointArrayLike, dst: PointArrayLike, *, uniform_scale: bool = True) -> tuple[Self, float]:
        src, dst = PointArray.as_array(src), PointArray.as_array(dst)
        diff = (dst - src).view(np.ndarray)
        t = Point.as_point(np.mean(diff, axis=0))
        return cls(t), float(np.mean(np.sum((diff - t.view(np.ndarray)) ** 2, axis=1)))

    def _warp[DTYPE: np.uint8 | np.float32](
        self, src_img: npt.NDArray[DTYPE], src_domain: IntRect, dst_domain: IntRect
    ) -> tuple[npt.NDArray[DTYPE], IntRect]:
        dst_img = dst_domain.crop_pad_image(src_img, origin=-src_domain.top_left, channel_last=True, copy=False)
        return dst_img, dst_domain
