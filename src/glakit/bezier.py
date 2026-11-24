from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Never, Self, overload

import numpy as np
import numpy.typing as npt

from .geometry import Point, PointArray, PointArrayLike, PointLike, _BasePointArray, _GeomOpsMixin, as_point, as_points
from .transform import Transform
from .typing import (
    Bool1DArray,
    Float1DArray,
    Float1DArrayLike,
    Float1DArrayLikeStrict,
    Float2DArray,
    FloatPairTuple,
    Int1DArray,
    as_float_1d,
)

type BezierCubicLike = npt.NDArray | tuple[PointLike, PointLike, PointLike, PointLike] | BezierCubic

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema


class BezierCubic(
    _GeomOpsMixin[
        "BezierCubic",
        "BezierCubic",
        Never,
        Never,
        "BezierCubic",
        "BezierCubic",
        PointArrayLike | float,
        PointArrayLike | float,
    ],
    _BasePointArray[Literal[4], np.float64, Point],
):
    _DTYPE = np.dtype(np.float64)
    _POINT_CLS = Point

    @overload
    def __new__(cls, bezier_cubic: BezierCubicLike, /) -> Self: ...
    @overload
    def __new__(cls, p0: PointLike, c0: PointLike, c1: PointLike, p1: PointLike) -> Self: ...
    def __new__(
        cls,
        p0: PointLike | BezierCubicLike,
        c0: PointLike | None = None,
        c1: PointLike | None = None,
        p1: PointLike | None = None,
    ) -> Self:
        if p1 is None:
            return cls.as_bcubic(p0)  # type: ignore
        if c0 is None:
            c0 = p0  # type: ignore
        if c1 is None:
            c1 = p1
        return cls.as_bcubic((p0, c0, c1, p1))  # type: ignore

    @classmethod
    def as_bcubic(cls, value: BezierCubicLike, copy: bool = True) -> Self:
        if isinstance(value, cls):
            return value.copy() if copy else value
        elif isinstance(value, np.ndarray):
            assert value.shape == (4, 2), "BezierCubic must be a (4, 2) array"
            return value.astype(np.float64, copy=copy).view(cls)
        elif isinstance(value, tuple):
            assert len(value) == 4, "BezierCubic must be a tuple of 4 points"
            return np.vstack([Point.as_point(_, copy=copy) for _ in value]).view(cls)
        else:
            raise TypeError(f"Cannot convert {type(value)} to BezierCubic")

    if TYPE_CHECKING:

        @overload
        def __getitem__(self, key: int | np.integer[Any]) -> Point: ...
        @overload
        def __getitem__(self, key: slice | Bool1DArray | Int1DArray | Sequence[int]) -> PointArray[Any]: ...
        @overload
        def __getitem__(self, key: tuple[int | np.integer[Any], int | np.integer[Any]]) -> float: ...
        @overload
        def __getitem__(
            self,
            key: tuple[slice | Bool1DArray | Int1DArray, int | np.integer[Any]] | tuple[int | np.integer[Any], slice],
        ) -> Float1DArray: ...
        @overload
        def __getitem__(self, key: tuple[slice | Bool1DArray | Int1DArray, slice]) -> Float2DArray: ...
        def __getitem__(self, key: Any) -> Any: ...  # type: ignore

    else:

        def __getitem__(self, key: Any) -> Any:
            result = super().__getitem__(key)
            if isinstance(result, BezierCubic) and result.shape != (4, 2):
                return result.view(PointArray)
            return result

    @classmethod
    def _wrap_ufunc_result(cls, res: Any) -> Any:
        wrapped = super()._wrap_ufunc_result(res)
        if isinstance(wrapped, PointArray) and wrapped.shape == (4, 2):
            return wrapped.view(cls)
        return wrapped

    @property
    def p0(self) -> Point:
        """The starting point of the Bezier curve."""
        return self[0]

    @property
    def c0(self) -> Point:
        """The first control point of the Bezier curve."""
        return self[1]

    @property
    def c1(self) -> Point:
        """The second control point of the Bezier curve."""
        return self[2]

    @property
    def p1(self) -> Point:
        """The ending point of the Bezier curve."""
        return self[3]

    @p0.setter
    def p0(self, value: Point) -> None:
        self[0] = value

    @c0.setter
    def c0(self, value: Point) -> None:
        self[1] = value

    @c1.setter
    def c1(self, value: Point) -> None:
        self[2] = value

    @p1.setter
    def p1(self, value: Point) -> None:
        self[3] = value

    @property
    def tuple(self) -> tuple[FloatPairTuple, FloatPairTuple, FloatPairTuple, FloatPairTuple]:
        """Return the Bezier curve as a tuple of points (p0, c0, c1, p1)."""
        return (self.p0.tuple, self.c0.tuple, self.c1.tuple, self.p1.tuple)

    @property
    def ndarray(self) -> np.ndarray[tuple[Literal[4], Literal[2]], np.dtype[np.float64]]:
        """Return the Bezier curve as a (4, 2) numpy array."""
        return self.view(np.ndarray)

    def __str__(self) -> str:
        return f"BezierCubic(p0={self.p0}, c0={self.c0}, c1={self.c1}, p1={self.p1})"

    def __repr__(self) -> str:
        return f"BezierCubic({self.tuple})"

    def __or__(self, other: BezierCubicLike) -> BSpline:  # type: ignore
        try:
            return BSpline([self, BezierCubic.as_bcubic(other)])
        except TypeError:
            return NotImplemented

    def __ror__(self, other: BezierCubicLike) -> BSpline:  # type: ignore
        try:
            return BSpline([BezierCubic.as_bcubic(other), self])
        except TypeError:
            return NotImplemented

    def to_path(self, offset: Point | None = None) -> str:
        if np.any(np.isnan(self.ndarray)):
            return ""
        bcubic = self
        if offset is not None:
            bcubic = self - offset
        path = f"M {bcubic.p0.x},{bcubic.p0.y} "
        if bcubic.p0 == bcubic.c0 and bcubic.c1 == bcubic.p1:
            path += f"L {bcubic.p1.x},{bcubic.p1.y}"
        else:
            path += f"C {bcubic.c0.x},{bcubic.c0.y} {bcubic.c1.x},{bcubic.c1.y} {bcubic.p1.x},{bcubic.p1.y}"
        return path

    def chord_length(self) -> float:
        return self.p0.distance(self.p1)

    def arc_length(self, fast_approximation=False):
        if fast_approximation:
            return (
                (self.p0 - self.p1).norm
                + (self.c0 - self.p0).norm
                + (self.c1 - self.c0).norm
                + (self.p1 - self.c1).norm
            ) / 2
        else:
            u = np.linspace(0, 1, int(self.arc_length(fast_approximation=True)) + 1, endpoint=True)
            return np.sum((self._q(u[1:]) - self._q(u[:-1])).norm)

    def evaluate(self, t: Float1DArrayLike) -> PointArray:
        return self._q(as_float_1d(t))

    def evaluate_tangent(self, t: Float1DArrayLike, normalized=False) -> PointArray:
        """Evaluate the tangent of the Bezier curve at a given parameter.

        Parameters
        ----------
        t : float
            The parameter at which to evaluate the tangent.
        normalized : bool, optional
            If True, the tangent is normalized. By default False.

        Returns
        -------
        PointArray
            The tangent of the Bezier curve at the given parameter.
        """
        tangents = self._qPrime(as_float_1d(t))
        if normalized:
            tangents = tangents.normalized()
        return tangents

    def flip(self) -> Self:
        return type(self)(self.p1, self.c1, self.c0, self.p0)

    def has_nan(self) -> bool:
        return bool(np.any(np.isnan(self)))

    def is_null(self) -> bool:
        return self.p0 == self.p1 and self.c0 == self.p0 and self.c1 == self.p1

    def c0_sym(self, d: float = -1, relative=True) -> Point:
        """Compute the symmetric control point of c0 with respect to p0.

        The argument d allows to move the symmetric control point along the line p0-c0.

        Parameters
        ----------
        d : int, optional
            Distance from c0 to the symmetric control point, by default -1 (i.e. symmetric of c0 with respect to p0).

        relative : bool, optional
            If True (by default), d is multiplied by the distance between c0 and p0.
            Otherwise d is considered as an absolute distance (in pixel).

        Returns
        -------
        Point
            The symmetric control point of c0 with respect to p0 or any point along the line p0-c0.
        """
        v = self.c0 - self.p0
        if not relative:
            v = v.normalized()
        return self.p0 + v * d

    def c1_sym(self, d: float = -1, relative=True) -> Point:
        """Compute the symmetric control point of c1 with respect to p1.

        The argument d allows to move the symmetric control point along the line p1-c1.

        Parameters
        ----------
        d : int, optional
            Distance from c1 to the symmetric control point, by default -1 (i.e. symmetric of c1 with respect to p1).

        relative : bool, optional
            If True (by default), d is multiplied by the distance between c1 and p1.
            Otherwise d is considered as an absolute distance (in pixel).

        Returns
        -------
        Point
            The symmetric control point of c1 with respect to p1 or any point along the line p1-c1.
        """
        v = self.c1 - self.p1
        if not relative:
            v = v.normalized()
        return self.p1 + v * d

    def split(self, t=0.5) -> tuple[BezierCubic, BezierCubic]:
        p01 = self.p0 + (self.c0 - self.p0) * t
        p12 = self.c0 + (self.c1 - self.c0) * t
        p23 = self.c1 + (self.p1 - self.c1) * t
        p012 = p01 + (p12 - p01) * t
        p123 = p12 + (p23 - p12) * t
        p0123 = p012 + (p123 - p012) * t
        return (BezierCubic(self.p0, p01, p012, p0123), BezierCubic(p0123, p123, p23, self.p1))

    @overload
    def projection(
        self,
        pts: PointArrayLike,
        fast_approximation: bool = False,
        *,
        return_u: Literal[False] = False,
        max_iteration: int = 20,
        error: float = 2,
    ) -> PointArray | tuple[PointArray, Float1DArray]: ...
    @overload
    def projection(
        self,
        pts: PointArrayLike,
        fast_approximation: bool = False,
        *,
        return_u: Literal[True],
        max_iteration: int = 20,
        error: float = 2,
    ) -> tuple[PointArray, Float1DArray]: ...
    def projection(
        self,
        pts: PointArrayLike,
        fast_approximation: bool = False,
        *,
        return_u: bool = False,
        max_iteration: int = 20,
        error: float = 2,
    ) -> PointArray | tuple[PointArray, Float1DArray]:
        """Project a set of points on the Bezier curve.

        Parameters
        ----------
        pts : np.ndarray
            The points to project on the Bezier curve.

        fast_approximation : bool, optional
                    If True, use a fast approximation by projecting the points on the trapezoid defined by the control points, by default False.

        return_u : bool, optional
            If True, also return the parameter u of the projected points on the Bezier curve, by default False.

        max_iteration : int, optional
            Maximum number of iterations of the Newton-Raphson algorithm, by default 20.
            If fast_approximation is True, this parameter is ignored.

        error : float, optional
            Target error in pixel for the projection stopping preemptively the Newton-Raphson algorithm, by default 2.
            If fast_approximation is True, this parameter is ignored.

        Returns
        -------
        projected_points : np.ndarray
            The projected points on the Bezier curve as a (n, 2) array.

        u : np.ndarray
            If return_u is True, also return the parameter u of the projected points on the Bezier curve as a (n,) array.
        """

        pts = as_points(pts)
        n = pts.shape[0]

        # Compute projection on the trapezoid defined by (p0, c0, c1, p1)
        p0, c0, c1, p1 = self

        if np.all(p0 == c0) and np.all(p0 == c1) and np.all(p0 == p1):
            proj = np.full_like(pts, p0).view(PointArray)
            return proj if not return_u else (proj, np.zeros(len(pts), dtype=np.float64))

        if np.any(p0 != c0):
            p0c0_u = ((c0 - p0) / (p0c0norm := np.linalg.norm(c0 - p0)))[None, :]
            l_p0c0 = np.clip((pts - p0) @ p0c0_u.T, 0, p0c0norm)
            p0c0_proj = l_p0c0 * p0c0_u + p0[None, :]
        else:
            p0c0_proj, l_p0c0, p0c0norm = p0[None, :], np.zeros((1, n)), 0
        if np.any(c1 != c0):
            c0c1_u = ((c1 - c0) / (c0c1norm := np.linalg.norm(c1 - c0)))[None, :]
            l_c0c1 = np.clip((pts - c0) @ c0c1_u.T, 0, c0c1norm)
            c0c1_proj = l_c0c1 * c0c1_u + c0[None, :]
        else:
            c0c1_proj, l_c0c1, c0c1norm = c0[None, :], np.zeros((1, n)), 0
        if np.any(p1 != c1):
            p1c1_u = ((p1 - c1) / (p1c1norm := np.linalg.norm(p1 - c1)))[None, :]
            l_p1c1 = np.clip((pts - c1) @ p1c1_u.T, 0, p1c1norm)
            p1c1_proj = l_p1c1 * p1c1_u + c1[None, :]
        else:
            p1c1_proj, l_p1c1, p1c1norm = c1[None, :], np.zeros((1, n)), 0

        proj = np.stack([p0c0_proj, c0c1_proj, p1c1_proj], axis=0)
        dist = np.linalg.norm(proj - pts[None, :, :], axis=2)
        closest_segment = np.argmin(dist, axis=0)

        if not fast_approximation or return_u:
            u = np.stack([l_p0c0, l_c0c1 + p0c0norm, l_p1c1 + p0c0norm + c0c1norm], axis=0).squeeze(axis=2)
            u = np.take_along_axis(u, closest_segment[None, :], axis=0).squeeze(axis=0)
            u /= p0c0norm + c0c1norm + p1c1norm
        else:
            u = np.zeros(n, dtype=np.float64)

        if fast_approximation:
            proj = np.take_along_axis(proj, closest_segment[None, :, None], axis=0).squeeze(axis=0).view(PointArray)
            return proj if not return_u else (proj, u)
        else:
            for _ in range(max_iteration):
                u = self._closestU_newtonRaphsonStep(pts, u)
                maxError = np.max((pts - self.evaluate(u)).sqr_norm)
                if maxError < error**2:
                    break
            u = np.clip(u, 0, 1)

            proj = self.evaluate(u)
            return proj if not return_u else (proj, u)

    def _q(self, u: Float1DArray) -> PointArray:
        u = u[:, np.newaxis]
        invU = 1.0 - u
        p0, c0, c1, p1 = self.ndarray[:, np.newaxis, :]
        q = invU**3 * p0 + 3 * invU**2 * u * c0 + 3 * invU * u**2 * c1 + u**3 * p1
        return q.view(PointArray)

    def _qPrime(self, u: Float1DArray) -> PointArray:
        u = u[:, np.newaxis]
        invU = 1.0 - u
        p0, c0, c1, p1 = self.ndarray[:, np.newaxis, :]
        qP = 3 * invU**2 * (c0 - p0) + 6 * invU * u * (c1 - c0) + 3 * u**2 * (p1 - c1)
        return qP.view(PointArray)

    def _qPrimePrime(self, u: Float1DArray) -> PointArray:
        u = u[:, np.newaxis]
        p0, c0, c1, p1 = self.ndarray[:, np.newaxis, :]
        qPP = (6 * (1.0 - u) * (c1 - 2 * c0 + p0) + 6 * u * (p1 - 2 * c1 + c0)).view(PointArray)
        return qPP

    def _closestU_newtonRaphsonStep(self, point: PointArray, u: Float1DArray, *, inplace: bool = False) -> Float1DArray:
        """
        Newton's root finding algorithm calculates f(x)=0 by reiterating
        x_n+1 = x_n - f(x_n)/f'(x_n)

        We are trying to find curve parameter u for some point p that minimizes
        the distance from that point to the curve. Distance point to curve is d=q(u)-p.
        At minimum distance the point is perpendicular to the curve.
        We are solving
        f = q(u)-p * q'(u) = 0
        with
        f' = q'(u) * q'(u) + q(u)-p * q''(u)

        gives
        u_n+1 = u_n - |q(u_n)-p * q'(u_n)| / |q'(u_n)**2 + q(u_n)-p * q''(u_n)|
        """
        if not inplace:
            u = u.copy()

        t = u[:, np.newaxis]
        invT = 1.0 - t
        p0, c0, c1, p1 = self.ndarray[:, np.newaxis, :]

        q = invT**3 * p0 + 3 * invT**2 * t * c0 + 3 * invT * t**2 * c1 + t**3 * p1
        qP = 3 * invT**2 * (c0 - p0) + 6 * invT * t * (c1 - c0) + 3 * t**2 * (p1 - c1)
        qPP = 6 * (1.0 - t) * (c1 - 2 * c0 + p0) + 6 * t * (p1 - 2 * c1 + c0)

        d = q - point
        num, den = (d * qP).sum(axis=1), (qP**2 + d * qPP).sum(axis=1)
        non_null_den = den != 0.0

        u[non_null_den] -= num[non_null_den] / den[non_null_den]
        return u

    ############################
    ##   Pydantic integration ##
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        from pydantic_core import core_schema

        def validate(v: Any) -> Self:
            if isinstance(v, cls):
                return v
            try:
                return cls.as_bcubic(v, copy=True)
            except TypeError as e:  # Cast TypeError to ValueError for Pydantic compatibility
                raise ValueError(str(e)) from e

        def serialize(v: Self) -> tuple[FloatPairTuple, FloatPairTuple, FloatPairTuple, FloatPairTuple]:
            return v.tuple

        py_schema = core_schema.no_info_plain_validator_function(validate)
        point_schema = core_schema.tuple_schema([core_schema.float_schema(), core_schema.float_schema()])
        bcubic_schema = core_schema.tuple_schema([point_schema] * 4)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(validate, bcubic_schema),
            python_schema=py_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize, return_schema=bcubic_schema),
        )


type BSplineLike = Iterable[BezierCubicLike] | BezierCubic | BSpline | str | npt.NDArray


class BSpline[BCubicType: BezierCubic](np.ndarray[tuple[int, 4, 2], np.dtype[np.float64]]):
    _BCUBIC_CLS: ClassVar[type[BezierCubic]] = BezierCubic

    def __new__(cls, bspline: BSplineLike = ()) -> Self:
        return cls.as_bspline(bspline, copy=False)

    @classmethod
    def as_bspline(cls, bspline: BSplineLike, copy: bool = True) -> Self:
        if isinstance(bspline, cls):
            return bspline.copy() if copy else bspline
        elif isinstance(bspline, BezierCubic):
            return (bspline.copy() if copy else bspline).ndarray[np.newaxis, :, :].view(cls)
        elif isinstance(bspline, np.ndarray):
            assert bspline.ndim == 3 and bspline.shape[1:] == (4, 2), "BSpline must be a (n, 4, 2) array"
            return bspline.astype(np.float64, copy=copy).view(cls)
        elif isinstance(bspline, str):
            return cls.parse_svg_path(bspline)
        elif isinstance(bspline, Iterable):
            curves = [BezierCubic.as_bcubic(curve, copy=copy) for curve in bspline]
            if not curves:
                return np.empty((0, 4, 2), dtype=np.float64).view(cls)
            return np.stack(curves, axis=0).view(cls)
        else:
            raise TypeError(f"Cannot convert {type(bspline)} to BSpline")

    @classmethod
    def parse_svg_path(cls, path_str: str) -> Self:
        """Parse an SVG path string into a BSpline."""
        svg = [s for s in path_str.replace(",", " ").split() if s]
        bspline = []
        p0 = Point.invalid()
        while svg:
            match command := svg.pop(0):
                case "M":
                    assert len(svg) >= 2, "Invalid SVG path: 'M' command requires two coordinates"
                    p0 = Point(x=float(svg.pop(0)), y=float(svg.pop(0)))
                case "L":
                    assert not p0.is_nan(), "Path must start with a 'M' command"
                    assert len(svg) >= 2, "Invalid SVG path: 'L' command requires two coordinates"
                    p1 = Point(x=float(svg.pop(0)), y=float(svg.pop(0)))
                    bspline.append(BezierCubic(p0, p0, p1, p1))
                    p0 = p1
                case "C":
                    assert not p0.is_nan(), "Path must start with a 'M' command"
                    assert len(svg) >= 6, "Invalid SVG path: 'C' command requires six coordinates"
                    c0 = Point(x=float(svg.pop(0)), y=float(svg.pop(0)))
                    c1 = Point(x=float(svg.pop(0)), y=float(svg.pop(0)))
                    p1 = Point(x=float(svg.pop(0)), y=float(svg.pop(0)))
                    bspline.append(BezierCubic(p0, c0, c1, p1))
                    p0 = p1
                case "S":
                    assert not p0.is_nan(), "Path must start with a 'M' command"
                    assert len(svg) >= 4, "Invalid SVG path: 'S' command requires four coordinates"
                    c1 = Point(x=float(svg.pop(0)), y=float(svg.pop(0)))
                    p1 = Point(x=float(svg.pop(0)), y=float(svg.pop(0)))
                    if bspline:
                        prev_curve = bspline[-1]
                        c0 = prev_curve.c1_sym(relative=True)
                    else:
                        c0 = p0
                    bspline.append(BezierCubic(p0, c0, c1, p1))
                    p0 = p1
                case _:
                    raise ValueError(f"Unsupported SVG path command: {command}")
        return cls(bspline)

    def __repr__(self) -> str:
        return f"BSpline([{','.join([repr(_) for _ in self])}])"

    def __str__(self) -> str:
        descr = f"BSpline({len(self)} curves):\n\t"
        return descr + "\n\t".join(str(curve) for curve in self)

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self) -> Iterator[BCubicType]:  # type: ignore
        for i in range(len(self)):
            yield self[i].view(self._BCUBIC_CLS)  # type: ignore

    @overload
    def __getitem__(self, key: slice | list[int]) -> Self: ...
    @overload
    def __getitem__(self, key: int | np.integer[Any]) -> BCubicType: ...
    @overload
    def __getitem__(self, key: tuple[int | np.integer[Any], int | np.integer[Any]]) -> Point: ...
    @overload
    def __getitem__(self, key: tuple[int | np.integer[Any], int | np.integer[Any], int | np.integer[Any]]) -> float: ...
    def __getitem__(self, key: Any) -> Any:  # type: ignore
        result = super().__getitem__(key)
        if np.isscalar(key):
            return result.view(self._BCUBIC_CLS)
        elif isinstance(key, tuple) and len(key) == 2 and np.isscalar(key[0]) and np.isscalar(key[1]):
            return result.copy().view(Point)
        elif isinstance(result, np.ndarray) and result.ndim == 3 and result.shape[1:] == (4, 2):
            return result.view(type(self))
        return result

    def to_path(
        self,
        offset: Point | None = None,
        mid_arrow: Literal["forward", "reverse"] | None = None,
        arrow_size: float = 10.0,
    ) -> str:
        """Convert the BSpline to a path string."""
        path = "\n".join(curve.to_path(offset) for curve in self)
        if mid_arrow is not None:
            u = self.relative_pos_to_t(0.5)
            p = self.evaluate(u)
            if offset is not None:
                p -= offset
            tan = self.evaluate_tangent(u, normalized=True) * arrow_size
            if mid_arrow == "reverse":
                tan = -tan
            left, right = p - tan.rotate(np.pi / 4), p - tan.rotate(-np.pi / 4)
            path += f"\nM {left.x:.2f} {left.y:.2f} L {p.x:.2f} {p.y:.2f} L {right.x:.2f} {right.y:.2f}"
        return path

    def __or__(self, other: Iterable[BezierCubicLike]) -> Self:  # type: ignore
        try:
            return type(self).as_bspline(itertools.chain(self, (BezierCubic.as_bcubic(_) for _ in other)))
        except TypeError:
            return NotImplemented

    def __ror__(self, other: Iterable[BezierCubicLike]) -> Self:  # type: ignore
        try:
            return type(self).as_bspline(itertools.chain((BezierCubic.as_bcubic(_) for _ in other), self))
        except TypeError:
            return NotImplemented

    def flip(self) -> Self:
        return self.__class__([curve.flip() for curve in reversed(self)])

    def filling_curves(
        self, start: PointLike | None = None, end: PointLike | None = None, *, smoothing: float = 0
    ) -> list[BCubicType]:
        filling = []
        if len(self) == 0:
            if start is not None and end is not None:
                filling.append(self._BCUBIC_CLS(start, start, end, end))
            return filling

        if start is not None:
            start = as_point(start)
            p_start = self[0].p0
            start_c = self[0].c0_sym(-start.distance(p_start) * smoothing, relative=False) if smoothing else p_start
            filling.append(self._BCUBIC_CLS(start, start, start_c, p_start))

        for prev, next in itertools.pairwise(self):
            if smoothing:
                dist = prev.p1.distance(next.p0)
                prev_c = prev.c1_sym(-dist * smoothing, relative=False)
                next_c = next.c0_sym(-dist * smoothing, relative=False)
            else:
                prev_c, next_c = prev.p1, next.p0
            filling.append(self._BCUBIC_CLS(prev.p1, prev_c, next_c, next.p0))

        if end is not None:
            end = as_point(end)
            p_last = self[-1].p1
            last_c = self[-1].c1_sym(-end.distance(p_last) * smoothing, relative=False) if smoothing else p_last
            filling.append(self._BCUBIC_CLS(p_last, last_c, end, end))

        return filling

    def interpolate(
        self, start: PointLike | None = None, end: PointLike | None = None, *, smoothing: float = 0
    ) -> Self:
        filling_curves = self.filling_curves(start, end, smoothing=smoothing)
        curves = list(self)

        bsplines = []
        if start is not None:
            bsplines.append(filling_curves.pop(0))
        for curve, filling in itertools.zip_longest(curves, filling_curves):
            bsplines.append(curve)
            if filling is not None:
                bsplines.append(filling)

        return self.as_bspline([b for b in bsplines if not b.is_null()])

    @overload
    def evaluate(self, t: float) -> Point: ...
    @overload
    def evaluate(self, t: Float1DArrayLikeStrict) -> PointArray: ...
    def evaluate(self, t: Float1DArrayLike) -> Point | PointArray:
        """Evaluate the position of the BSpline for a set of parameters t.

        Parameters
        ----------
        t : Float1DArrayLike
            The parameters to evaluate the BSpline at. Each parameter corresponds to a position on the BSpline, where the integer part is the index of the Bezier curve and the fractional part is the relative position on that Bezier curve.

        Returns
        -------
        Point | PointArray
            The evaluated points on the BSpline as a (n, 2) array if t is a vector, or as a (2,) array if t is scalar.
        """
        is_single = np.isscalar(t)
        if len(self) == 0:
            return Point.invalid() if is_single else np.empty((0, 2), dtype=float).view(PointArray)
        t = as_float_1d(t)

        points = np.empty((len(t), 2), dtype=float)
        curves_id = np.clip(np.floor(t).astype(int), 0, len(self) - 1)
        u = t - curves_id
        for c in np.unique(curves_id):
            mask = curves_id == c
            points[mask] = self[c].evaluate(u[mask])

        return Point.as_point(points[0]) if is_single else points.view(PointArray)

    @overload
    def evaluate_tangent(self, t: float, normalized=False) -> Point: ...
    @overload
    def evaluate_tangent(self, t: Float1DArrayLikeStrict, normalized=False) -> PointArray: ...
    def evaluate_tangent(self, t: Float1DArrayLike, normalized=False) -> Point | PointArray:
        """Evaluate the tangent of the BSpline for a set of parameters t.

        Parameters
        ----------
        t : Float1DArrayLike
            The parameters to evaluate the BSpline at. Each parameter corresponds to a position on the BSpline, where the integer part is the index of the Bezier curve and the fractional part is the relative position on that Bezier curve.

        normalized : bool, optional
            If True, return normalized tangents, by default False.

        Returns
        -------
        Point | PointArray
            The evaluated tangents on the BSpline as a (n, 2) array if t is a vector, or as a (2,) array if t is scalar.
        """
        is_single = np.isscalar(t)
        if len(self) == 0:
            return Point.invalid() if is_single else np.empty((0, 2), dtype=float).view(PointArray)
        t = as_float_1d(t)
        assert t.ndim == 1, "t must be a 1D array"

        tangents = np.empty((len(t), 2), dtype=float)
        curves_id = np.clip(np.floor(t).astype(int), 0, len(self) - 1)
        u = t - curves_id
        for c in np.unique(curves_id):
            mask = curves_id == c
            tangents[mask] = self[c].evaluate_tangent(u[mask], normalized=normalized)

        return Point.as_point(tangents[0]) if is_single else tangents.view(PointArray)

    @overload
    def relative_pos_to_t(self, pos: float, fast_approximation=False) -> float: ...
    @overload
    def relative_pos_to_t(self, pos: Float1DArrayLikeStrict, fast_approximation=False) -> Float1DArray: ...
    def relative_pos_to_t(self, pos: Float1DArrayLike, fast_approximation=False) -> float | Float1DArray:
        """
        Convert relative positions on the BSpline to the corresponding parameter t.

        Parameters
        ----------
        pos : Float1DArrayLike
            The relative positions on the BSpline (between 0 and 1).

        fast_approximation : bool, optional
            If True, use a fast approximation for the arc length calculation. By default False.

        Returns
        -------
        float | Float1DArray
            The corresponding parameter t on the BSpline.
        """
        is_single = np.isscalar(pos)
        if len(self) == 0:
            return float("nan") if is_single else np.empty((0,), dtype=float)
        pos = as_float_1d(pos)

        lengths = np.array([curve.arc_length(fast_approximation=fast_approximation) for curve in self])
        lengths = np.concatenate(([0], np.cumsum(lengths)))
        if lengths[-1] == 0:
            lengths[:] = 0
        else:
            lengths /= lengths[-1]  # Normalize to [0, 1]

        curve_ids = np.clip(np.searchsorted(lengths, pos, side="right") - 1, 0, len(self) - 1)
        curve_start = lengths[curve_ids]
        curve_end = lengths[curve_ids + 1]
        t = curve_ids + (pos - curve_start) / (curve_end - curve_start + 1e-10)

        return t[0] if is_single else t

    def tips_tangents(self, normalize=True) -> tuple[Point, Point]:
        t0 = self[0].c0 - self[0].p0
        t1 = self[-1].c1 - self[-1].p1
        return (t0.normalized(), t1.normalized()) if normalize else (t0, t1)

    def chord_length(self) -> float:
        if len(self) == 0:
            return 0
        return self[0].p0.distance(self[-1].p1)

    def arc_length(self, fast_approximation=False):
        return sum(curve.arc_length(fast_approximation) for curve in self)

    def transform(self, projection: Transform, *, round_p: bool = False) -> Self:
        bspline_data = self.view(np.ndarray).reshape(-1, 2)
        bspline_data = projection.transform(bspline_data).reshape(-1, 4, 2)
        if round_p:
            t0 = bspline_data[:, 1, :] - bspline_data[:, 0, :]
            t1 = bspline_data[:, 2, :] - bspline_data[:, 3, :]
            bspline_data[:, 0, :] = np.round(bspline_data[:, 0, :])
            bspline_data[:, 3, :] = np.round(bspline_data[:, 3, :])
            bspline_data[:, 1, :] = bspline_data[:, 0, :] + t0
            bspline_data[:, 2, :] = bspline_data[:, 3, :] + t1
        return bspline_data.view(type(self))

    @overload
    def projection(
        self, points: PointArrayLike, fast_approximation: bool = False, *, return_u: Literal[False] = False
    ) -> PointArray: ...
    @overload
    def projection(
        self, points: PointArrayLike, fast_approximation: bool = False, *, return_u: Literal[True]
    ) -> tuple[PointArray, Float1DArray]: ...

    def projection(
        self, points: PointArrayLike, fast_approximation: bool = False, *, return_u: bool = False
    ) -> PointArray | tuple[PointArray, Float1DArray]:
        """Project a set of points on the Bezier curve.

        Parameters
        ----------
        yx_points : PointArray
            The points to project on the Bezier curve.
        fast_approximation : bool, optional
            If True, use a fast approximation by projecting the points on the trapezoid defined by the control points, by default False.
        return_u : bool, optional
            If True, also return the parameter u of the projected points on the Bezier curve, by default False

        Returns
        -------
        projected_points : PointArray
            The projected points on the Bezier curve as a (n, 2) array.

        u : np.ndarray
            If return_u is True, also return the parameter u of the projected points on the Bezier curve as a (n,) array. ``floor(u)`` is the index of the Bezier curve in the BSpline, and ``u - floor(u)`` is the relative position on the Bezier curve.

        """
        points = as_points(points)
        n_split = points.shape[0]

        if len(self) == 0:
            proj = np.full((n_split, 2), np.nan, dtype=np.float64).view(PointArray)
            return proj if not return_u else (proj, np.full(n_split, np.nan, dtype=np.float64))

        # === Find the closest bezier curve for each split point ===
        projs, u = zip(
            *[bezier.projection(points, return_u=True, fast_approximation=fast_approximation) for bezier in self],
            strict=True,
        )
        if len(self) == 1:
            proj = projs[0]
            return proj if not return_u else (proj, u[0])
        else:
            dists = [(projs[i] - points).sqr_norm for i in range(len(self))]
            closest_bezier = np.argmin(dists, axis=0)
            proj = (
                np.take_along_axis(np.array(projs), closest_bezier[None, :, None], axis=0)
                .squeeze(axis=0)
                .view(PointArray)
            )
            if return_u:
                u = np.take_along_axis(np.array(u), closest_bezier[None, :], axis=0).squeeze(axis=0)
                u += closest_bezier
                return proj, u
            else:
                return proj

    def split_into_multiple_bsplines(
        self, split_coord: PointArrayLike | None = None, u: Float1DArrayLike | None = None
    ) -> list[BSpline]:
        if u is None:
            if split_coord is None:
                raise ValueError("Either split_coord or u must be provided")
            else:
                _, u = self.projection(split_coord, return_u=True, fast_approximation=False)
        else:
            u = as_float_1d(u)

        closest_bezier = np.floor(u).astype(int)
        u = u - closest_bezier
        previous_end = (u == 0) & (closest_bezier > 0)
        closest_bezier[previous_end] -= 1
        u[previous_end] = 1

        # === Split the closest bezier curve at the closest point ===
        beziers = [[]]
        for i, bezier in enumerate(self):
            if i not in closest_bezier:
                beziers[-1].append(bezier)
                continue

            # Select the split points and their parameter u for the current bezier curve
            # split_points, split_u = split_coord[closest_bezier == i], u[closest_bezier == i]
            split_u = u[closest_bezier == i]

            # Remove points that are at the extremities of the bezier curve
            p0_points, p1_points = np.isclose(split_u, 0), np.isclose(split_u, 1)
            not_extremities = ~(p0_points | p1_points)
            # split_points, split_u = split_points[not_extremities], split_u[not_extremities]
            split_u = split_u[not_extremities]

            # Refine the parameter u and remove any additional extremities
            # split_u = np.clip(np.sort(bezier.parametrize(split_points, initial_u=split_u)), 0, 1)
            # p0_points, p1_points = np.isclose(split_u, 0), np.isclose(split_u, 1)
            # not_extremities = ~(p0_points | p1_points)
            # split_points, split_u = split_points[not_extremities], split_u[not_extremities]

            # Create empty segment for any points before the bezier curve
            for _ in range(np.count_nonzero(p0_points)):
                beziers.append([])

            # Split the bezier curve at the split points
            if len(split_u):
                acc_t = 0
                for t in split_u:
                    b0, bezier = bezier.split((t - acc_t) / (1 - acc_t))
                    acc_t += t
                    # Append the beginning of the bezier curve (before the split) to the current segment ...
                    beziers[-1].append(b0)
                    # ... and start a new segment
                    beziers.append([])

            # Append the remaining bezier curve to the current segment
            beziers[-1].append(bezier)

            # Create empty segment for any points after the bezier curve
            for _ in range(np.count_nonzero(p1_points)):
                beziers.append([])

        return [BSpline(_) for _ in beziers]

    def split(self, split_coord: PointArrayLike | None = None, u: Float1DArrayLike | None = None) -> BSpline:
        return BSpline(itertools.chain(*self.split_into_multiple_bsplines(split_coord, u)))

    ############################
    ##   Pydantic integration ##
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        from pydantic_core import core_schema

        def validate(v: Any) -> Self:
            if isinstance(v, cls):
                return v
            try:
                return cls.as_bspline(v, copy=True)
            except TypeError as e:  # Cast TypeError to ValueError for Pydantic compatibility
                raise ValueError(str(e)) from e

        def serialize(v: Self) -> list[tuple[FloatPairTuple, FloatPairTuple, FloatPairTuple, FloatPairTuple]]:
            return [bcubic.tuple for bcubic in v]

        py_schema = core_schema.no_info_plain_validator_function(validate)
        point_schema = core_schema.tuple_schema([core_schema.float_schema(), core_schema.float_schema()])
        bspline_schema = core_schema.list_schema(core_schema.tuple_schema([point_schema] * 4))

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(validate, bspline_schema),
            python_schema=py_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize, return_schema=bspline_schema),
        )
