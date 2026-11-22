from __future__ import annotations

import functools
import math
from collections.abc import Iterable, Iterator, Sequence
from types import EllipsisType, GenericAlias
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Never, Self, overload

import numpy as np
import numpy.typing as npt

from . import if_none
from .typing import (
    Bool1DArray,
    Float1DArray,
    FloatPair,
    FloatPairArray,
    FloatPairTuple,
    Int1DArray,
    IntPair,
    IntPairArray,
    IntPairTuple,
)

_INT_MAX = int(np.iinfo(np.int64).max)
_INT_MIN = int(np.iinfo(np.int64).min)


if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import CoreSchema

    class _GeomOpsMixin[
        SelfT,
        FloatSelfT,
        IntRectT,
        FloatRectT,
        IntArrayT,
        FloatArrayT,
        IntOtherT,
        FloatOtherT,
    ]:
        """Unified static typing mixin enforcing arithmetic precedence:
        IntPoint < Point < IntRect < Rect < IntPointArray < PointArray.
        """

        @overload
        def __add__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __add__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __add__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __add__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __add__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __add__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __add__(self, other: Any) -> Any: ...

        @overload
        def __radd__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __radd__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __radd__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __radd__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __radd__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __radd__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __radd__(self, other: Any) -> Any: ...

        @overload
        def __sub__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __sub__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __sub__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __sub__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __sub__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __sub__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __sub__(self, other: Any) -> Any: ...

        @overload
        def __rsub__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __rsub__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __rsub__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __rsub__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __rsub__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __rsub__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __rsub__(self, other: Any) -> Any: ...

        @overload
        def __mul__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __mul__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __mul__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __mul__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __mul__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __mul__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __mul__(self, other: Any) -> Any: ...

        @overload
        def __rmul__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __rmul__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __rmul__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __rmul__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __rmul__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __rmul__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __rmul__(self, other: Any) -> Any: ...

        @overload
        def __floordiv__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __floordiv__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __floordiv__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __floordiv__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __floordiv__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __floordiv__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __floordiv__(self, other: Any) -> Any: ...

        @overload
        def __rfloordiv__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __rfloordiv__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __rfloordiv__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __rfloordiv__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __rfloordiv__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __rfloordiv__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __rfloordiv__(self, other: Any) -> Any: ...

        @overload
        def __mod__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __mod__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __mod__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __mod__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __mod__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __mod__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __mod__(self, other: Any) -> Any: ...

        @overload
        def __rmod__(self, other: IntRectT) -> IntRectT: ...
        @overload
        def __rmod__(self, other: FloatRectT) -> FloatRectT: ...
        @overload
        def __rmod__(self, other: IntArrayT) -> IntArrayT: ...
        @overload
        def __rmod__(self, other: FloatArrayT) -> FloatArrayT: ...
        @overload
        def __rmod__(self, other: IntOtherT) -> SelfT: ...
        @overload
        def __rmod__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __rmod__(self, other: Any) -> Any: ...

        @overload
        def __truediv__(self, other: IntRectT | FloatRectT) -> FloatRectT: ...
        @overload
        def __truediv__(self, other: IntArrayT | FloatArrayT) -> FloatArrayT: ...
        @overload
        def __truediv__(self, other: IntOtherT | FloatOtherT) -> FloatSelfT: ...
        def __truediv__(self, other: Any) -> Any: ...

        @overload
        def __rtruediv__(self, other: IntRectT | FloatRectT) -> FloatRectT: ...
        @overload
        def __rtruediv__(self, other: IntArrayT | FloatArrayT) -> FloatArrayT: ...
        @overload
        def __rtruediv__(self, other: IntOtherT | FloatOtherT) -> FloatSelfT: ...
        def __rtruediv__(self, other: Any) -> Any: ...

        def __neg__(self) -> Self: ...
        def __pos__(self) -> Self: ...
        def __abs__(self) -> Self: ...

    class _PointIndexMixin[CoordT, SliceArrT]:
        """Static typing mixin for Point and IntPoint indexing.
        CoordT: The type of the coordinate values (e.g., float, int, etc.).
        SliceArrT: The type of the array returned when slicing (e.g., Float1DArray, Int1DArray, etc.).
        """

        @overload
        def __getitem__(self, key: int | np.integer[Any]) -> CoordT: ...
        @overload
        def __getitem__(self, key: slice | Bool1DArray | Int1DArray) -> SliceArrT: ...
        def __getitem__(self, key: Any) -> Any: ...

    class _PointArrayMixin[PointT, SelfT, PointLikeT, ArrayLikeT, CoordT, ColArrT, MatArrT]:
        """Static typing mixin for PointArray and IntPointArray indexing and iteration."""

        @overload
        def __getitem__(self, key: int | np.integer[Any]) -> PointT: ...
        @overload
        def __getitem__(
            self, key: slice | Bool1DArray | Int1DArray | Sequence[int] | EllipsisType | tuple[EllipsisType, slice]
        ) -> SelfT: ...
        @overload
        def __getitem__(self, key: tuple[int | np.integer[Any], int | np.integer[Any]]) -> CoordT: ...
        @overload
        def __getitem__(
            self,
            key: tuple[slice | Bool1DArray | Int1DArray | EllipsisType, int | np.integer[Any]]
            | tuple[int | np.integer[Any], slice | EllipsisType],
        ) -> ColArrT: ...
        @overload
        def __getitem__(self, key: tuple[slice | Bool1DArray | Int1DArray, slice | EllipsisType]) -> MatArrT: ...
        def __getitem__(self, key: Any) -> Any: ...

        @overload
        def __setitem__(self, key: int | np.integer[Any], value: PointLikeT) -> None: ...
        @overload
        def __setitem__(
            self,
            key: slice | Bool1DArray | Int1DArray | Sequence[int] | EllipsisType | tuple[EllipsisType, slice],
            value: ArrayLikeT | PointLikeT,
        ) -> None: ...
        @overload
        def __setitem__(
            self,
            key: tuple[
                int | np.integer[Any] | slice | Bool1DArray | Int1DArray | EllipsisType,
                int | np.integer[Any] | slice | EllipsisType,
            ],
            value: CoordT | ColArrT | MatArrT,
        ) -> None: ...
        def __setitem__(self, key: Any, value: Any) -> None: ...

        def __iter__(self) -> Iterator[PointT]: ...

    class _PointMapMixin[PointT, RowArrayT, SelfT, PointLikeT, ArrayLikeT, MapLikeT, CoordT, Field2DArrT]:
        """Static typing mixin for PointMap, IntPointMap, and VectorMap indexing and reductions."""

        @overload
        def __getitem__(self, key: int | np.integer[Any]) -> RowArrayT: ...
        @overload
        def __getitem__(
            self, key: slice | Bool1DArray | Int1DArray | Sequence[int] | EllipsisType | tuple[EllipsisType, slice]
        ) -> SelfT: ...
        @overload
        def __getitem__(self, key: tuple[int | np.integer[Any], int | np.integer[Any]]) -> PointT: ...
        @overload
        def __getitem__(
            self,
            key: tuple[slice | Bool1DArray | Int1DArray, int | np.integer[Any]]
            | tuple[int | np.integer[Any], slice | Bool1DArray | Int1DArray],
        ) -> RowArrayT: ...
        @overload
        def __getitem__(
            self, key: tuple[slice | Bool1DArray | Int1DArray, slice | Bool1DArray | Int1DArray | EllipsisType]
        ) -> SelfT: ...
        @overload
        def __getitem__(
            self, key: tuple[int | np.integer[Any], int | np.integer[Any], int | np.integer[Any]]
        ) -> CoordT: ...
        @overload
        def __getitem__(
            self, key: tuple[EllipsisType, int | np.integer[Any]] | tuple[Any, Any, int | np.integer[Any]]
        ) -> Field2DArrT: ...
        def __getitem__(self, key: Any) -> Any: ...

        @overload
        def __setitem__(self, key: tuple[int | np.integer[Any], int | np.integer[Any]], value: PointLikeT) -> None: ...
        @overload
        def __setitem__(self, key: int | np.integer[Any], value: ArrayLikeT | PointLikeT) -> None: ...
        @overload
        def __setitem__(
            self,
            key: slice
            | EllipsisType
            | tuple[slice | Bool1DArray | Int1DArray | EllipsisType, slice | Bool1DArray | Int1DArray | EllipsisType],
            value: MapLikeT | ArrayLikeT | PointLikeT | CoordT,
        ) -> None: ...
        @overload
        def __setitem__(
            self,
            key: tuple[EllipsisType, int | np.integer[Any]]
            | tuple[Any, Any, int | np.integer[Any]]
            | tuple[int | np.integer[Any], int | np.integer[Any], int | np.integer[Any]],
            value: Field2DArrT | RowArrayT | CoordT,
        ) -> None: ...
        def __setitem__(self, key: Any, value: Any) -> None: ...

    class _RectIndexMixin[PointT, CoordT]:
        """Static typing mixin for Rect and IntRect indexing.
        PointT: The type of the point returned when indexing (e.g., Point, IntPoint, etc.).
        CoordT: The type of the coordinate values (e.g., float, int, etc.).
        """

        @overload
        def __getitem__(self, key: int | np.integer[Any]) -> PointT: ...
        @overload
        def __getitem__(self, key: tuple[int | np.integer[Any], int | np.integer[Any]]) -> CoordT: ...
        @overload
        def __getitem__(self, key: Any) -> Any: ...
        def __getitem__(self, key: Any) -> Any: ...

    class _RectSetOpsMixin[IntOtherT, FloatOtherT, FloatSelfT]:
        """Static typing mixin for Rect and IntRect intersection (&) and union (|)."""

        @overload
        def __and__(self, other: IntOtherT) -> Self: ...
        @overload
        def __and__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __and__(self, other: Any) -> Any: ...

        @overload
        def __rand__(self, other: IntOtherT) -> Self: ...
        @overload
        def __rand__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __rand__(self, other: Any) -> Any: ...

        @overload
        def __or__(self, other: IntOtherT) -> Self: ...
        @overload
        def __or__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __or__(self, other: Any) -> Any: ...

        @overload
        def __ror__(self, other: IntOtherT) -> Self: ...
        @overload
        def __ror__(self, other: FloatOtherT) -> FloatSelfT: ...
        def __ror__(self, other: Any) -> Any: ...

else:

    class _GeomOpsMixin[SelfT, FloatSelfT, IntRectT, FloatRectT, IntArrayT, FloatArrayT, IntOtherT, FloatOtherT]:
        pass

    class _FloatOpsMixin[OtherT]:
        pass

    class _IntOpsMixin[IntOtherT, FloatOtherT, FloatSelfT]:
        pass

    class _PointIndexMixin[CoordT, SliceArrT]:
        pass

    class _PointArrayMixin[PointT, SelfT, PointLikeT, ArrayLikeT, CoordT, ColArrT, MatArrT]:
        pass

    class _PointMapMixin[PointT, RowArrayT, SelfT, PointLikeT, ArrayLikeT, MapLikeT, CoordT, Field2DArrT]:
        pass

    class _RectIndexMixin[PointT, CoordT]:
        pass

    class _RectSetOpsMixin[IntOtherT, FloatOtherT, FloatSelfT]:
        pass


class _GenericArrayMixin:
    """Ensure Generic.__class_getitem__ takes precedence over np.ndarray.__class_getitem__."""

    def __class_getitem__(cls, params: Any) -> Any:
        return GenericAlias(cls, params)


_DTYPE_ORDER: dict[np.dtype[Any], int] = {
    np.dtype(np.int64): 0,
    np.dtype(np.float32): 1,
    np.dtype(np.float64): 2,
}


def maximal_dtype(dtypes: Iterable[npt.DTypeLike], default: np.dtype[Any] | None) -> np.dtype[Any]:
    """Return the highest encompassing dtype among (np.int64 < np.float32 < np.float64) from an iterable of dtypes."""
    return max(
        (_normalize_dtype(dt) for dt in dtypes),
        default=if_none(default, np.dtype(np.int64)),
        key=_DTYPE_ORDER.__getitem__,
    )


def _normalize_dtype(dtype: npt.DTypeLike) -> np.dtype[Any]:
    """Map any NumPy dtype-like to the supported hierarchy (np.int64 < np.float32 < np.float64)."""
    dt = np.dtype(dtype)
    if dt in _DTYPE_ORDER:
        return dt
    if np.issubdtype(dt, np.integer):
        return np.dtype(np.int64)
    if np.issubdtype(dt, np.floating):
        return np.dtype(np.float32) if dt.itemsize <= 4 else np.dtype(np.float64)
    return np.dtype(np.float64)


def minimal_dtype(obj: Any, min_dtype: np.dtype[Any] | None = None) -> np.dtype[Any]:
    """Return the highest encompassing dtype among (np.int64 < np.float32 < np.float64) needed to encode `obj`.

    `min_dtype` sets both the lower bound and the default floating precision for pure Python literals.
    """
    min_dtype_ = if_none(min_dtype, np.dtype(np.int64))

    def ensure_min_dtype(candidate: npt.DTypeLike) -> np.dtype[Any]:
        return maximal_dtype((candidate, min_dtype_), default=min_dtype_)

    default_float = ensure_min_dtype(min_dtype_ if np.issubdtype(min_dtype_, np.floating) else np.dtype(np.float64))

    match obj:
        case np.dtype():
            return ensure_min_dtype(obj)

        case _BasePoint() | _BasePointArray() | _BasePointMap() | _BaseRect():
            if np.issubdtype(obj._DTYPE, np.floating) and np.all(np.equal(np.mod(obj.view(np.ndarray), 1), 0)):
                return min_dtype_
            return ensure_min_dtype(obj._DTYPE)

        case np.ndarray():
            if np.issubdtype(obj.dtype, np.floating) and np.all(np.equal(np.mod(obj, 1), 0)):
                return min_dtype_
            return ensure_min_dtype(obj.dtype)

        case np.floating() if float(obj).is_integer():
            return min_dtype_

        case np.integer() | np.floating():
            return ensure_min_dtype(type(obj))

        case bool() | int():
            return min_dtype_

        case float():
            return min_dtype_ if obj.is_integer() else default_float

        case str():
            return default_float if ("." in obj or "e" in obj.lower()) else min_dtype_

        case Sequence():
            return maximal_dtype((minimal_dtype(item, min_dtype_) for item in obj), default=min_dtype_)

        case _:
            return min_dtype_


#######################################################
#    === POINTS: _BasePoint, Point, IntPoint) ===
#######################################################

type _PointInputs = "FloatPairTuple | FloatPair | IntPairTuple | IntPair | str | npt.NDArray | float"


class _BasePoint[DTypeT: np.floating[Any] | np.integer[Any], ItemT: float | int](
    _GenericArrayMixin, np.ndarray[tuple[Literal[2]], np.dtype[DTypeT]]
):
    """Base immutable 2D point class parameterized by NumPy scalar dtype."""

    _DTYPE: ClassVar[np.dtype[Any]]

    @classmethod
    def _ITEM_TYPE(cls) -> type[ItemT]:
        raise NotImplementedError

    @classmethod
    @functools.cache
    def origin(cls) -> Self:
        raise NotImplementedError

    def __new__(cls, y: _PointInputs | Self | float, x: float | None = None) -> Self:
        if x is None:
            return cls.as_point(y, copy=True)
        if not np.isscalar(y) or not np.isscalar(x):
            raise TypeError("If a second argument is provided to Point(y, x), both arguments must be scalars")
        arr = np.array((cls._as_item(y), cls._as_item(x)), dtype=cls._DTYPE)
        arr.flags.writeable = False
        return arr.view(cls)

    @classmethod
    def by_dtype(cls, dtype: npt.DTypeLike) -> type[_BasePoint[Any, Any]]:
        dt = np.dtype(dtype)
        return IntPoint if np.issubdtype(dt, np.integer) else Point

    @classmethod
    def as_point(cls, s: Any, copy: bool = False) -> Self:
        if isinstance(s, cls) and not copy:
            return s
        if isinstance(s, str):
            try:
                s = s.strip("()")
                y_str, x_str = s.split(",")
                y, x = cls._as_item(y_str), cls._as_item(x_str)
            except (ValueError, TypeError):
                raise TypeError(f"Invalid 2d point string: {s}") from None
        elif isinstance(s, (tuple, list)) and len(s) == 2 and all(np.isscalar(_) for _ in s):
            y, x = cls._as_item(s[0]), cls._as_item(s[1])
        elif isinstance(s, np.ndarray) and s.shape == (2,):
            arr = s.astype(cls._DTYPE, copy=True)
            arr.flags.writeable = False
            return arr.view(cls)
        elif np.isscalar(s) and not isinstance(s, str):
            y = x = cls._as_item(s)
        else:
            raise TypeError(f"Invalid 2d point: {s}")
        arr = np.array([y, x], dtype=cls._DTYPE)
        arr.flags.writeable = False
        return arr.view(cls)

    @classmethod
    def _as_item(cls, item: Any) -> ItemT:
        return cls._ITEM_TYPE()(item)

    @classmethod
    def _wrap_as_point(cls, arr: np.ndarray) -> Self:
        pt = arr.astype(cls._DTYPE, copy=False)
        pt.flags.writeable = False
        return pt.view(cls)

    @classmethod
    def _wrap_ufunc_result(cls, res: Any) -> Any:
        # Promote to float Point if ufunc produced floats or if called on float Point
        if isinstance(res, np.ndarray) and res.shape == (2,) and np.issubdtype(res.dtype, np.number):
            if np.issubdtype(res.dtype, np.floating):
                return Point._wrap_as_point(res)
            return cls._wrap_as_point(res)
        return res

    def __array_finalize__(self, obj):
        if obj is not None and np.shares_memory(self, obj):
            self.flags.writeable = False

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if any(isinstance(item, (_BasePointArray, _BasePointMap)) for item in inputs):
            return NotImplemented  # Defer to PointArray/PointMap.__array_ufunc__
        if "out" in kwargs:
            raise TypeError("Point objects are immutable and do not support in-place operations ('out' parameter)")

        args = []
        for item in inputs:
            if isinstance(item, _BasePoint):
                args.append(item.view(np.ndarray))
            elif isinstance(item, np.ndarray) or (np.isscalar(item) and not isinstance(item, str)):
                args.append(item)
            else:
                dt = minimal_dtype(item)
                pt_cls = _BasePoint.by_dtype(dt)
                args.append(pt_cls.as_point(item, copy=False).view(np.ndarray))

        result = getattr(ufunc, method)(*args, **kwargs)

        if result is NotImplemented:
            return NotImplemented
        if isinstance(result, tuple):
            return tuple(self._wrap_ufunc_result(x) for x in result)
        return self._wrap_ufunc_result(result)

    def copy(self, order: Literal["C", "F", "A", "K"] | None = None) -> np.ndarray[Any, np.dtype[DTypeT]]:
        return self.view(np.ndarray).copy(order=order)

    if not TYPE_CHECKING:

        def __getitem__(self, key):
            result = super().__getitem__(key)
            if isinstance(result, _BasePoint) and result.shape != (2,):
                return result.view(np.ndarray)
            return result

    def __eq__(self, other: object) -> bool:
        if np.isscalar(other) and not isinstance(other, str):
            return False
        try:
            other_pt = other if isinstance(other, _BasePoint) else Point.as_point(other)
            return bool(np.allclose(self.view(np.ndarray), other_pt.view(np.ndarray)))
        except (ValueError, TypeError):
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:  # type: ignore
        arr = self.view(np.ndarray)
        return hash((float(arr[0]), float(arr[1])))

    def as_int(self) -> IntPoint:
        """Convert the point to an IntPoint by rounding the coordinates."""
        arr = self.view(np.ndarray)
        return IntPoint(round(float(arr[0])), round(float(arr[1])))

    def ceil(self) -> IntPoint:
        return IntPoint(math.ceil(self[0]), math.ceil(self[1]))

    def floor(self) -> IntPoint:
        return IntPoint(math.floor(self[0]), math.floor(self[1]))

    def as_float(self) -> Point:
        arr = self.view(np.ndarray)
        return Point(float(arr[0]), float(arr[1]))

    @property
    def y(self) -> ItemT:
        """The y-coordinate of the point."""
        return self._as_item(self.view(np.ndarray)[0])

    @property
    def x(self) -> ItemT:
        """The x-coordinate of the point."""
        return self._as_item(self.view(np.ndarray)[1])

    @property
    def tuple(self) -> tuple[ItemT, ItemT]:
        p = self.view(np.ndarray)
        return (self._as_item(p[0]), self._as_item(p[1]))

    @property
    def xy(self) -> tuple[ItemT, ItemT]:
        p = self.view(np.ndarray)
        return (self._as_item(p[1]), self._as_item(p[0]))

    @property
    def ndarray(self) -> np.ndarray[tuple[Literal[2]], np.dtype[DTypeT]]:
        return self.view(np.ndarray)

    @property
    def norm(self) -> float:
        arr = self.view(np.ndarray)
        return math.hypot(float(arr[0]), float(arr[1]))

    @property
    def sqr_norm(self) -> float:
        arr = self.view(np.ndarray)
        return float(arr[0]) ** 2 + float(arr[1]) ** 2

    @property
    def angle(self) -> float:
        arr = self.view(np.ndarray)
        return math.atan2(float(arr[0]), float(arr[1]))

    def normalized(self) -> Point:
        n = self.norm
        return Point.origin() if n == 0 else (self / n).view(Point)

    def rotate(self, angle: float, mode: Literal["deg", "rad"] = "rad") -> Point:
        angle_rad = math.radians(angle) if mode == "deg" else angle
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        y, x = self.view(np.ndarray)
        return Point(y * cos_a - x * sin_a, y * sin_a + x * cos_a)

    def rot90(self) -> Self:
        y, x = self.view(np.ndarray)
        return type(self)(-x, y)

    @overload
    def distance(self, other: AnyPointLikeStrict) -> float: ...
    @overload
    def distance(self, other: PointArrayLikeStrict) -> Float1DArray: ...
    def distance(self, other: PointArrayLike) -> float | Float1DArray:
        try:
            other_pt = Point.as_point(other, copy=False).view(np.ndarray)
            diff = other_pt - self.view(np.ndarray)
            return float(np.hypot(diff[0], diff[1]))
        except (ValueError, TypeError):
            try:
                other_arr = PointArray.as_array(other, copy=False).view(np.ndarray)
                diff = other_arr - self.view(np.ndarray)
                return np.hypot(diff[:, 0], diff[:, 1])
            except (ValueError, TypeError):
                raise TypeError("distance() expects a PointLike or PointArrayLike input")

    ############################
    ##   Pydantic integration ##
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        from pydantic_core import core_schema

        def validate(value: Any) -> Self:
            if isinstance(value, cls):
                return value
            try:
                return cls.as_point(value, copy=True)
            except TypeError as e:  # Cast TypeError to ValueError for Pydantic compatibility
                raise ValueError(str(e)) from e

        item_schema = core_schema.float_schema() if np.issubdtype(cls._DTYPE, np.floating) else core_schema.int_schema()
        point_schema = core_schema.tuple_schema([item_schema, item_schema])
        py_schema = core_schema.no_info_plain_validator_function(validate)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                validate,
                core_schema.union_schema([point_schema, core_schema.str_schema()]),
            ),
            python_schema=py_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda pt: pt.tuple,
                return_schema=point_schema,
            ),
        )


class Point(
    _GeomOpsMixin[
        "Point",
        "Point",
        "Rect",
        "Rect",
        "PointArray[Any]",
        "PointArray[Any]",
        "PointLike | float",
        "PointLike | float",
    ],
    _PointIndexMixin[float, Float1DArray],
    _BasePoint[np.float64, float],
):
    """Immutable 2D floating-point Point (y, x)."""

    _DTYPE = np.dtype(np.float64)

    @classmethod
    def _ITEM_TYPE(cls) -> type[float]:
        return float

    @classmethod
    @functools.cache
    def origin(cls) -> Point:
        return cls(0.0, 0.0)

    @classmethod
    @functools.cache
    def invalid(cls) -> Point:
        return cls(np.nan, np.nan)

    def __str__(self) -> str:
        return f"({self.y:.1f}, {self.x:.1f})"

    def __repr__(self) -> str:
        return f"Point(y={self.y:.1f}, x={self.x:.1f})"

    def is_nan(self) -> bool:
        return math.isnan(self.y) or math.isnan(self.x)

    def as_float(self) -> Point:
        return self

    def clip_to(self, rect: RectLike) -> Point:
        r = Rect.as_rect(rect)
        return Point(min(max(self.y, r.top), r.bottom), min(max(self.x, r.left), r.right))

    def cross(self, other: PointLike, normalize: bool = False) -> float:
        other_pt = Point.as_point(other)
        p = self.normalized() if normalize else self
        if normalize:
            other_pt = other_pt.normalized()
        return p.x * other_pt.y - p.y * other_pt.x

    def point_dot(self, other: PointLike, normalize: bool = False) -> float:
        other_pt = Point.as_point(other)
        p = self.normalized() if normalize else self
        if normalize:
            other_pt = other_pt.normalized()
        return p.x * other_pt.x + p.y * other_pt.y


class IntPoint(
    _GeomOpsMixin[
        "IntPoint",
        "Point",
        "IntRect",
        "Rect",
        "IntPointArray[Any]",
        "PointArray[Any]",
        "IntPointLike | int",
        "PointLike | float",
    ],
    _PointIndexMixin[int, Int1DArray],
    _BasePoint[np.int64, int],
):
    """Immutable 2D integer Point (y, x)."""

    _DTYPE = np.dtype(np.int64)

    @classmethod
    def _ITEM_TYPE(cls) -> type[int]:
        return int

    @classmethod
    @functools.cache
    def origin(cls) -> IntPoint:
        return cls(0, 0)

    @classmethod
    def as_point(cls, s: Any, copy: bool = False) -> Self:
        # Reject non-integer floats during implicit ufunc casting so IntPoint + (0.5, 1.5) promotes to Point
        if np.issubdtype(minimal_dtype(s), np.floating):
            raise TypeError("Non-integer values cannot be cast losslessly to IntPoint")
        return super().as_point(s, copy=copy)

    def as_int(self) -> IntPoint:
        return self

    def __str__(self) -> str:
        return f"({self.y}, {self.x})"

    def __repr__(self) -> str:
        return f"IntPoint(y={self.y}, x={self.x})"

    def rot90(self) -> IntPoint:
        return IntPoint(-self.x, self.y)


as_point = Point.as_point
as_int_point = IntPoint.as_point


#############################################################################
#    === POINT ARRAYS: _BasePointArray, PointArray, IntPointArray ===
#############################################################################
class _BasePointArray[
    N: int,
    DTypeT: np.floating[Any] | np.integer[Any],
    PointT: _BasePoint[Any, Any],
](_GenericArrayMixin, np.ndarray[tuple[N, Literal[2]], np.dtype[DTypeT]]):
    """Base 2D point array parameterized by length N, NumPy dtype, and Point subclass."""

    _DTYPE: ClassVar[np.dtype[Any]]
    _POINT_CLS: ClassVar[type[_BasePoint[Any, Any]]]

    def __new__(cls, points: Any, copy: bool = True) -> Self:
        return cls.as_array(points, copy=copy)

    @classmethod
    def by_dtype(cls, dtype: npt.DTypeLike) -> type[_BasePointArray[Any, Any, Any]]:
        return IntPointArray if np.issubdtype(np.dtype(dtype), np.integer) else PointArray

    @classmethod
    def as_array(cls, s: Any, copy: bool = False) -> Self:
        if isinstance(s, cls):
            out = s.astype(cls._DTYPE, copy=copy)
        elif isinstance(s, np.ndarray):
            if s.ndim == 2 and s.shape[1] == 2:
                out = s.astype(cls._DTYPE, copy=copy)
            elif s.shape == (2,):
                out = s.astype(cls._DTYPE, copy=True).reshape(1, 2)
            elif s.size == 0:
                out = np.empty((0, 2), dtype=cls._DTYPE)
            else:
                raise ValueError(f"Expected ndarray of shape (N, 2), got {s.shape}")

        elif isinstance(s, (_BasePoint, str)) or (
            isinstance(s, tuple) and len(s) == 2 and all(np.isscalar(_) for _ in s)
        ):
            pt = cls._POINT_CLS.as_point(s, copy=False).view(np.ndarray)
            out = pt.reshape(1, 2)

        elif isinstance(s, Sequence):
            if len(s) == 0:
                return np.empty((0, 2), dtype=cls._DTYPE).view(cls)
            rows = [cls._POINT_CLS.as_point(item, copy=False).view(np.ndarray) for item in s]
            out = np.vstack(rows).astype(cls._DTYPE, copy=False)
        else:
            raise TypeError(f"Invalid PointArray input: {s}")

        if not out.flags.writeable or copy:
            out = out.copy()
            out.flags.writeable = True
        return out.view(cls)

    @classmethod
    def _wrap_ufunc_result(cls, res: Any) -> Any:
        if isinstance(res, np.ndarray) and np.issubdtype(res.dtype, np.number):
            target_dtype = minimal_dtype(res.dtype, cls._DTYPE)
            if res.ndim == 2 and res.shape[1] == 2:
                target_cls = cls.by_dtype(target_dtype)
                return res.astype(target_cls._DTYPE, copy=False).view(target_cls)
            if res.shape == (2,):
                target_pt = _BasePoint.by_dtype(target_dtype)
                return target_pt._wrap_as_point(res)
        return res

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if (
            any(isinstance(x, _BasePointMap) for x in inputs)
            or not isinstance(self, _BaseRect)
            and any(isinstance(x, _BaseRect) for x in inputs)
        ):
            return NotImplemented

        args = []
        for x in inputs:
            if isinstance(x, (_BasePointArray, _BasePoint)):
                args.append(x.view(np.ndarray))
            elif isinstance(x, np.ndarray) or (np.isscalar(x) and not isinstance(x, str)):
                args.append(x)
            else:
                dt = minimal_dtype(x)
                pt_cls = _BasePoint.by_dtype(dt)
                ar_cls = _BasePointArray.by_dtype(dt)
                try:
                    args.append(pt_cls.as_point(x, copy=False).view(np.ndarray))
                except (ValueError, TypeError):
                    try:
                        args.append(ar_cls.as_array(x, copy=False).view(np.ndarray))
                    except (ValueError, TypeError):
                        args.append(x)

        if "out" in kwargs:
            kwargs["out"] = tuple(
                x.view(np.ndarray) if isinstance(x, (_BasePointArray, _BasePoint)) else x for x in kwargs["out"]
            )

        result = getattr(ufunc, method)(*args, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        if isinstance(result, tuple):
            return tuple(self._wrap_ufunc_result(x) for x in result)
        return self._wrap_ufunc_result(result)

    if not TYPE_CHECKING:

        def __getitem__(self, key):
            result = super().__getitem__(key)
            if isinstance(result, np.ndarray) and result.shape == (2,) and not isinstance(key, tuple):
                return self._POINT_CLS._wrap_as_point(result.copy())
            if isinstance(result, _BasePointArray) and (result.ndim != 2 or result.shape[1] != 2):
                return result.view(np.ndarray)
            return result

        def __setitem__(self, key, value):
            try:
                value = self.as_array(value, copy=False).view(np.ndarray)
            except (ValueError, TypeError):
                pass
            self.view(np.ndarray)[key] = value

    if TYPE_CHECKING:

        @overload
        def mean(self, axis: Literal[0]) -> Point: ...
        @overload
        def mean(self, axis: Literal[1]) -> np.ndarray[tuple[N], np.dtype[DTypeT]]: ...
        @overload
        def mean(self, axis: None = None) -> np.floating[Any]: ...
        def mean(  # pyright: ignore[reportIncompatibleMethodOverride]
            self, axis: Any = None, dtype: Any = None, out: Any = None, keepdims: bool = False, *, where: Any = ...
        ) -> Any: ...

    @property
    def y(self) -> np.ndarray[tuple[N], np.dtype[DTypeT]]:
        return self.view(np.ndarray)[:, 0]

    @property
    def x(self) -> np.ndarray[tuple[N], np.dtype[DTypeT]]:
        return self.view(np.ndarray)[:, 1]

    @property
    def ndarray(self) -> np.ndarray[tuple[N, Literal[2]], np.dtype[DTypeT]]:
        return self.view(np.ndarray)

    def __iter__(self) -> Iterator[PointT]:  # type: ignore
        for i in range(len(self)):
            yield self._POINT_CLS._wrap_as_point(self.view(np.ndarray)[i].copy())  # type: ignore

    def __str__(self) -> str:
        return f"[{', '.join(str(p) for p in self)}]"

    @overload
    def distance(self, other: AnyPointLikeStrict) -> Float1DArray: ...
    @overload
    def distance(self, other: PointArrayLikeStrict) -> FloatPairArray: ...
    def distance(self, other: PointArrayLike) -> Float1DArray | FloatPairArray:
        try:
            other_pt = Point.as_point(other, copy=False).view(np.ndarray)
            diff = self.view(np.ndarray) - other_pt
            return np.hypot(diff[:, 0], diff[:, 1])
        except (ValueError, TypeError):
            try:
                other_arr = PointArray.as_array(other, copy=False).view(np.ndarray)
                diff = self.view(np.ndarray)[:, np.newaxis, :] - other_arr[np.newaxis, :, :]
                return np.hypot(diff[..., 0], diff[..., 1])
            except (ValueError, TypeError):
                raise TypeError("distance() expects a PointLike or PointArrayLike input")

    def angle(self) -> Float1DArray:
        arr = self.view(np.ndarray)
        return np.arctan2(arr[:, 0], arr[:, 1])

    @property
    def norm(self) -> Float1DArray:
        arr = self.view(np.ndarray)
        return np.hypot(arr[:, 0], arr[:, 1])

    @property
    def sqr_norm(self) -> np.ndarray[tuple[N], np.dtype[np.float64]]:
        arr = self.view(np.ndarray)
        return arr[:, 0] ** 2 + arr[:, 1] ** 2

    def normalized(self) -> PointArray[N]:
        n = self.norm
        nonzero_mask = n > 0
        arr = self.view(np.ndarray).copy()
        arr[nonzero_mask] /= n[nonzero_mask][:, np.newaxis]
        return arr.view(PointArray)

    def as_int(self) -> _BasePointArray[N, np.int64, IntPoint]:
        return np.rint(self.view(np.ndarray)).astype(np.int64).view(IntPointArray)

    def as_float(self) -> _BasePointArray[N, np.float64, Point]:
        return self.view(np.ndarray).astype(np.float64, copy=True).view(PointArray)

    ##############################
    ##   Pydantic integration ##
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        from pydantic_core import core_schema

        def validate(value: Any) -> Self:
            if isinstance(value, cls):
                return value
            try:
                return cls.as_array(value, copy=True)
            except TypeError as e:  # Cast TypeError to ValueError for Pydantic compatibility
                raise ValueError(str(e)) from e

        item_schema = core_schema.float_schema() if np.issubdtype(cls._DTYPE, np.floating) else core_schema.int_schema()
        array_schema = core_schema.list_schema(core_schema.tuple_schema([item_schema, item_schema]))
        py_schema = core_schema.no_info_plain_validator_function(validate)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(validate, array_schema),
            python_schema=py_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda arr: [p.tuple for p in arr],
                return_schema=array_schema,
            ),
        )


class PointArray[N: int](
    _GeomOpsMixin[
        "PointArray[N]",
        "PointArray[N]",
        Never,
        Never,
        "PointArray[N]",
        "PointArray[N]",
        "PointArrayLike | float",
        "PointArrayLike | float",
    ],
    _PointArrayMixin[Point, "PointArray[N]", "PointLike", "PointArrayLike", float, Float1DArray, FloatPairArray],
    _BasePointArray[N, np.float64, Point],
):
    """Mutable 2D floating-point array of shape (N, 2)."""

    _DTYPE = np.dtype(np.float64)
    _POINT_CLS = Point


class IntPointArray[N: int](
    _GeomOpsMixin[
        "IntPointArray[N]",
        "PointArray[N]",
        Never,
        Never,
        "IntPointArray[N]",
        "PointArray[N]",
        "IntPointArrayLike | IntPointLike | int",
        "PointArrayLike | float",
    ],
    _PointArrayMixin[IntPoint, "IntPointArray[N]", "IntPointLike", "IntPointArrayLike", int, Int1DArray, IntPairArray],
    _BasePointArray[N, np.int64, IntPoint],
):
    """Mutable 2D integer array of shape (N, 2)."""

    _DTYPE = np.dtype(np.int64)
    _POINT_CLS = IntPoint

    @classmethod
    def as_array(cls, s: Any, copy: bool = False) -> Self:
        # Reject non-integer floats during implicit ufunc casting so IntPointArray + (0.5, 1.5) promotes to PointArray
        if np.issubdtype(minimal_dtype(s), np.floating):
            raise TypeError("Non-integer values cannot be cast losslessly to IntPointArray")
        return super().as_array(s, copy=copy)

    if not TYPE_CHECKING:

        def __setitem__(self, key, value):
            if np.issubdtype(minimal_dtype(value), np.floating):
                raise TypeError("Cannot assign non-integer values to IntPointArray")
            super().__setitem__(key, value)


as_points = PointArray.as_array
as_int_points = IntPointArray.as_array


#############################################################################
#    === POINT MAPS: _BasePointMap, PointMap, IntPointMap ===
#############################################################################
class _BasePointMap[
    H: int,
    W: int,
    DTypeT: np.floating[Any] | np.integer[Any],
    PointT: _BasePoint[Any, Any],
    ArrayPointT: _BasePointArray[Any, Any, Any],
](_GenericArrayMixin, np.ndarray[tuple[H, W, Literal[2]], np.dtype[DTypeT]]):
    """Mutable 2D vector field of shape (H, W, 2) parameterized by height H, width W, NumPy dtype, and Point subclass."""

    _DTYPE: ClassVar[np.dtype[Any]]
    _POINT_CLS: ClassVar[type[_BasePoint[Any, Any]]]
    _ARRAY_CLS: ClassVar[type[_BasePointArray[Any, Any, Any]]]

    def __new__(cls, points: Any, copy: bool = True) -> Self:
        return cls.as_map(points, copy=copy)

    @classmethod
    def by_dtype(cls, dtype: npt.DTypeLike) -> type[_BasePointMap[Any, Any, Any, Any, Any]]:
        dt = np.dtype(dtype)
        if dt == np.dtype(np.float32):
            return VectorMap
        else:
            return IntPointMap if np.issubdtype(dt, np.integer) else PointMap

    @classmethod
    def zeros(cls, shape: tuple[int, int] | IntPointLike) -> Self:
        h, w = IntPoint.as_point(shape, copy=False).tuple
        return np.zeros((h, w, 2), dtype=cls._DTYPE).view(cls)

    @classmethod
    def as_map(cls, s: Any, copy: bool = False) -> Self:
        if isinstance(s, cls):
            out = s.astype(cls._DTYPE, copy=copy)
        elif isinstance(s, np.ndarray):
            if s.ndim == 3 and s.shape[2] == 2:
                out = s.astype(cls._DTYPE, copy=copy)
            elif s.size == 0 and s.ndim == 3 and s.shape[2] == 2:
                out = np.empty(s.shape, dtype=cls._DTYPE)
            else:
                raise ValueError(f"Expected ndarray of shape (H, W, 2), got {s.shape}")
        elif isinstance(s, Sequence) and not isinstance(s, str):
            if len(s) == 0:
                return np.empty((0, 0, 2), dtype=cls._DTYPE).view(cls)
            rows = [cls._ARRAY_CLS.as_array(row, copy=False).view(np.ndarray) for row in s]
            out = np.stack(rows, axis=0).astype(cls._DTYPE, copy=False)
        else:
            raise TypeError(f"Invalid PointMap input: {s}")

        if not out.flags.writeable or copy:
            out = out.copy()
            out.flags.writeable = True
        return out.view(cls)

    @classmethod
    def _wrap_ufunc_result(cls, res: Any) -> Any:
        if isinstance(res, np.ndarray) and np.issubdtype(res.dtype, np.number):
            target_dtype = minimal_dtype(res.dtype, cls._DTYPE)
            if res.ndim == 3 and res.shape[2] == 2:
                target_cls = cls.by_dtype(target_dtype)
                return res.astype(target_cls._DTYPE, copy=False).view(target_cls)
            if res.ndim == 2 and res.shape[1] == 2:
                target_arr = _BasePointArray.by_dtype(target_dtype)
                return res.astype(target_arr._DTYPE, copy=False).view(target_arr)
            if res.shape == (2,):
                target_pt = _BasePoint.by_dtype(target_dtype)
                return target_pt._wrap_as_point(res)
        return res

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        target_dtype = maximal_dtype((minimal_dtype(x, self._DTYPE) for x in inputs), default=self._DTYPE)

        args = []
        for x in inputs:
            if isinstance(x, (_BasePointMap, _BasePointArray, _BasePoint)):
                args.append(x.view(np.ndarray))
            elif isinstance(x, (int, float)) and not isinstance(x, bool):
                args.append(target_dtype.type(x) if target_dtype == np.dtype(np.float32) else x)
            elif isinstance(x, np.ndarray) or (np.isscalar(x) and not isinstance(x, str)):
                args.append(x)
            else:
                dt = minimal_dtype(x, self._DTYPE)
                pt_cls = _BasePoint.by_dtype(dt)
                ar_cls = _BasePointArray.by_dtype(dt)
                mp_cls = _BasePointMap.by_dtype(dt)
                try:
                    args.append(pt_cls.as_point(x, copy=False).view(np.ndarray).astype(dt, copy=False))
                except (ValueError, TypeError):
                    try:
                        args.append(ar_cls.as_array(x, copy=False).view(np.ndarray).astype(dt, copy=False))
                    except (ValueError, TypeError):
                        try:
                            args.append(mp_cls.as_map(x, copy=False).view(np.ndarray).astype(dt, copy=False))
                        except (ValueError, TypeError):
                            args.append(x)

        if "out" in kwargs:
            kwargs["out"] = tuple(
                x.view(np.ndarray) if isinstance(x, (_BasePointMap, _BasePointArray, _BasePoint)) else x
                for x in kwargs["out"]
            )

        result = getattr(ufunc, method)(*args, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        if isinstance(result, tuple):
            return tuple(self._wrap_ufunc_result(x) for x in result)
        return self._wrap_ufunc_result(result)

    if not TYPE_CHECKING:

        def __getitem__(self, key):
            result = super().__getitem__(key)
            if not isinstance(result, np.ndarray):
                return result
            is_spatial_only = not isinstance(key, tuple) or (Ellipsis not in key and len(key) <= 2)
            if result.shape == (2,) and is_spatial_only:
                return self._POINT_CLS._wrap_as_point(result.copy())
            if result.ndim == 2 and result.shape[1] == 2 and is_spatial_only:
                return result.view(self._ARRAY_CLS)
            if isinstance(result, _BasePointMap) and (result.ndim != 3 or result.shape[2] != 2):
                return result.view(np.ndarray)
            return result

        def __setitem__(self, key, value):
            for converter in (self._POINT_CLS.as_point, self._ARRAY_CLS.as_array, self.as_map):
                try:
                    value = converter(value, copy=False).view(np.ndarray)
                    break
                except (ValueError, TypeError):
                    pass
            self.view(np.ndarray)[key] = value

    @property
    def y(self) -> np.ndarray[tuple[H, W], np.dtype[DTypeT]]:
        return self.view(np.ndarray)[..., 0]

    @property
    def x(self) -> np.ndarray[tuple[H, W], np.dtype[DTypeT]]:
        return self.view(np.ndarray)[..., 1]

    @property
    def ndarray(self) -> np.ndarray[tuple[H, W, Literal[2]], np.dtype[DTypeT]]:
        return self.view(np.ndarray)

    @property
    def norm(self) -> np.ndarray[tuple[H, W], np.dtype[np.float64]]:
        arr = self.view(np.ndarray)
        return np.hypot(arr[..., 0], arr[..., 1])

    @property
    def sqr_norm(self) -> np.ndarray[tuple[H, W], np.dtype[DTypeT]]:
        arr = self.view(np.ndarray)
        return arr[..., 0] ** 2 + arr[..., 1] ** 2

    def angle(self) -> np.ndarray[tuple[H, W], np.dtype[np.float64]]:
        arr = self.view(np.ndarray)
        return np.arctan2(arr[..., 0], arr[..., 1])

    def normalized(self) -> PointMap[H, W]:
        n = self.norm
        nonzero_mask = n > 0
        arr = self.view(np.ndarray).astype(np.float64, copy=True)
        arr[nonzero_mask] /= n[nonzero_mask][..., np.newaxis]
        return arr.view(PointMap)

    def as_array(self) -> ArrayPointT:
        return self.view(np.ndarray).reshape(-1, 2).view(self._ARRAY_CLS)  # type: ignore

    def as_int(self) -> IntPointMap[H, W]:
        return np.rint(self.view(np.ndarray)).astype(np.int64).view(IntPointMap)

    def as_float(self) -> PointMap[H, W]:
        return self.view(np.ndarray).astype(np.float64, copy=False).view(PointMap)

    def as_vector(self) -> VectorMap[H, W]:
        return self.view(np.ndarray).astype(np.float32, copy=False).view(VectorMap)

    ##############################
    ##   Pydantic integration ##
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        from pydantic_core import core_schema

        def validate(value: Any) -> Self:
            if isinstance(value, cls):
                return value
            try:
                return cls.as_map(value, copy=True)
            except TypeError as e:
                raise ValueError(str(e)) from e

        item_schema = core_schema.float_schema() if np.issubdtype(cls._DTYPE, np.floating) else core_schema.int_schema()
        map_schema = core_schema.list_schema(
            core_schema.list_schema(core_schema.tuple_schema([item_schema, item_schema]))
        )
        py_schema = core_schema.no_info_plain_validator_function(validate)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(validate, map_schema),
            python_schema=py_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda m: [[tuple(cls._POINT_CLS._as_item(c) for c in pt) for pt in row] for row in m.view(np.ndarray)],
                return_schema=map_schema,
            ),
        )


class PointMap[H: int, W: int](
    _GeomOpsMixin[
        "PointMap[H, W]",
        "PointMap[H, W]",
        Never,
        Never,
        "PointMap[H, W]",
        "PointMap[H, W]",
        "PointMapLike | PointArrayLike | float",
        "PointMapLike | PointArrayLike | float",
    ],
    _PointMapMixin[
        Point,
        PointArray[W],
        "PointMap[H, W]",
        "PointLike",
        "PointArrayLike",
        "PointMapLike",
        float,
        np.ndarray[tuple[H, W], np.dtype[np.float64]],
    ],
    _BasePointMap[H, W, np.float64, Point, PointArray[W]],
):
    """Mutable 2D floating-point vector field of shape (H, W, 2)."""

    _DTYPE = np.dtype(np.float64)
    _POINT_CLS = Point
    _ARRAY_CLS = PointArray

    def as_float(self) -> PointMap[H, W]:
        return self


class IntPointMap[H: int, W: int](
    _GeomOpsMixin[
        "IntPointMap[H, W]",
        "PointMap[H, W]",
        Never,
        Never,
        "IntPointMap[H, W]",
        "PointMap[H, W]",
        "IntPointMapLike | IntPointArrayLike | IntPointLike | int",
        "PointMapLike | PointArrayLike | float",
    ],
    _PointMapMixin[
        IntPoint,
        IntPointArray[W],
        "IntPointMap[H, W]",
        "IntPointLike",
        "IntPointArrayLike",
        "IntPointMapLike",
        int,
        np.ndarray[tuple[H, W], np.dtype[np.int64]],
    ],
    _BasePointMap[H, W, np.int64, IntPoint, IntPointArray[W]],
):
    """Mutable 2D integer vector field of shape (H, W, 2)."""

    _DTYPE = np.dtype(np.int64)
    _POINT_CLS = IntPoint
    _ARRAY_CLS = IntPointArray

    @classmethod
    def as_map(cls, s: Any, copy: bool = False) -> Self:
        if np.issubdtype(minimal_dtype(s), np.floating):
            raise TypeError("Non-integer float array cannot be cast losslessly to IntPointMap")
        return super().as_map(s, copy=copy)

    def as_int(self) -> IntPointMap[H, W]:
        return self

    if not TYPE_CHECKING:

        def __setitem__(self, key, value):
            if np.issubdtype(minimal_dtype(value), np.floating):
                raise TypeError("Cannot assign floating-point values to IntPointMap")
            super().__setitem__(key, value)


if TYPE_CHECKING:
    _Float32LiteralLike = (
        FloatPairTuple
        | IntPairTuple
        | list[FloatPairTuple]
        | list[IntPairTuple]
        | Sequence[Sequence[FloatPairTuple]]
        | Sequence[Sequence[IntPairTuple]]
        | FloatPairArray
        | IntPairArray
        | float
        | int
    )


class VectorMap[H: int, W: int](
    _GeomOpsMixin[
        "VectorMap[H, W]",
        "PointMap[H, W]",
        Never,
        Never,
        "VectorMap[H, W]",
        "PointMap[H, W]",
        "VectorMap[Any, Any] | npt.NDArray[np.float32] | _Float32LiteralLike",
        "PointMapLike | PointArrayLike | PointLike",
    ],
    _PointMapMixin[
        Point,
        PointArray[W],
        "VectorMap[H, W]",
        "PointLike",
        "PointArrayLike",
        "VectorMapLike",
        float,
        np.ndarray[tuple[H, W], np.dtype[np.float32]],
    ],
    _BasePointMap[H, W, np.float32, Point, PointArray[W]],
):
    """Mutable 2D single-precision (float32) vector field of shape (H, W, 2)."""

    _DTYPE = np.dtype(np.float32)
    _POINT_CLS = Point
    _ARRAY_CLS = PointArray

    @property
    def norm(self) -> np.ndarray[tuple[H, W], np.dtype[np.float32]]:  # type: ignore[override]
        arr = self.view(np.ndarray)
        return np.hypot(arr[..., 0], arr[..., 1])

    def angle(self) -> np.ndarray[tuple[H, W], np.dtype[np.float32]]:  # type: ignore[override]
        arr = self.view(np.ndarray)
        return np.arctan2(arr[..., 0], arr[..., 1])

    def normalized(self) -> VectorMap[H, W]:  # type: ignore[override]
        n = self.norm
        nonzero_mask = n > 0
        arr = self.view(np.ndarray).copy()
        arr[nonzero_mask] /= n[nonzero_mask][..., np.newaxis]
        return arr.view(VectorMap)

    def as_vector(self) -> VectorMap[H, W]:
        return self

    def dot_prod(self, other: VectorMapLike) -> np.ndarray[tuple[H, W], np.dtype[np.float32]]:
        """Element-wise dot product with another vector field or vector."""
        other_arr = VectorMap.as_map(other, copy=False).view(np.ndarray)
        arr = self.view(np.ndarray)
        return arr[..., 0] * other_arr[..., 0] + arr[..., 1] * other_arr[..., 1]

    def cross(self, other: VectorMapLike) -> np.ndarray[tuple[H, W], np.dtype[np.float32]]:
        """Element-wise 2D cross product (determinant: x1*y2 - y1*x2)."""
        other_arr = VectorMap.as_map(other, copy=False).view(np.ndarray)
        arr = self.view(np.ndarray)
        return arr[..., 1] * other_arr[..., 0] - arr[..., 0] * other_arr[..., 1]


as_vector_map = VectorMap.as_map


as_point_map = PointMap.as_map
as_int_point_map = IntPointMap.as_map
as_vector_map = VectorMap.as_map

##########################################################
#    === RECTANGLES: Rect, IntRect ===
##########################################################


class _BaseRect[
    DTypeT: np.floating[Any] | np.integer[Any],
    ItemT: float | int,
    PointT: _BasePoint[Any, Any],
    PointArrayT: _BasePointArray[Any, Any, Any],
](_BasePointArray[Literal[2], DTypeT, _BasePoint[DTypeT, ItemT]]):
    """Base immutable 2D rectangle class parameterized by NumPy dtype, Point, and PointArray subclasses."""

    _ARRAY_CLS: ClassVar[type[_BasePointArray[Any, Any, Any]]]

    def __new__(cls, rect_or_p0: RectLike | PointLike, p1: PointLike | None = None) -> Self:
        if p1 is not None:
            tl_pt = cls._POINT_CLS.as_point(rect_or_p0, copy=False)
            br_pt = cls._POINT_CLS.as_point(p1, copy=False)
            return cls._from_points(tl_pt, br_pt)
        return cls.as_rect(rect_or_p0, copy=True)

    @classmethod
    def by_dtype(cls, dtype: npt.DTypeLike) -> type[_BaseRect[Any, Any, Any, Any]]:
        return IntRect if np.issubdtype(np.dtype(dtype), np.integer) else Rect

    @classmethod
    @functools.cache
    def empty(cls) -> Self:
        raise NotImplementedError

    @classmethod
    def as_rect(cls, s: RectLike | PointLike, *, copy: bool = False) -> Self:
        if isinstance(s, cls):
            return (s.ndarray.copy() if copy else s).view(cls)

        # Try parsing as a single PointLike -> Rect((0, 0), s)
        try:
            pt = cls._POINT_CLS.as_point(s, copy=False)
            origin = cls._POINT_CLS.as_point((0, 0), copy=False)
            return cls._from_points(origin, pt)
        except (ValueError, TypeError):
            pass

        if isinstance(s, np.ndarray):
            if s.shape == (2, 2):
                return cls._from_points(cls._POINT_CLS.as_point(s[0]), cls._POINT_CLS.as_point(s[1]))
            if s.shape == (4,):
                return cls._from_points(cls._POINT_CLS.as_point(s[:2]), cls._POINT_CLS.as_point(s[2:]))
            raise ValueError(f"Expected ndarray of shape (2, 2) or (4,), got {s.shape}")

        if isinstance(s, Sequence) and not isinstance(s, str):
            if len(s) == 2:
                return cls._from_points(cls._POINT_CLS.as_point(s[0]), cls._POINT_CLS.as_point(s[1]))
            if len(s) == 4 and all(np.isscalar(_) for _ in s):
                as_item = cls._POINT_CLS._as_item
                return cls._from_points(
                    cls._POINT_CLS(as_item(s[0]), as_item(s[1])),
                    cls._POINT_CLS(as_item(s[2]), as_item(s[3])),
                )

        raise TypeError(f"Invalid rect input: {s}")

    def is_empty(self) -> bool:
        raise NotImplementedError

    @classmethod
    def _from_points(cls, p0: _BasePoint[DTypeT, ItemT], p1: _BasePoint[DTypeT, ItemT]) -> Self:
        as_item = cls._POINT_CLS._as_item
        p0_ = p0.view(np.ndarray)
        p1_ = p1.view(np.ndarray)
        y0, x0 = as_item(p0_[0]), as_item(p0_[1])
        y1, x1 = as_item(p1_[0]), as_item(p1_[1])
        arr = np.array([[y0, x0], [y1, x1]], dtype=cls._DTYPE)
        arr.flags.writeable = False
        return arr.view(cls)

    @staticmethod
    def _flatten_rects(*rects: Iterable[RectLike] | RectLike) -> list[_BaseRect]:
        flat: tuple[RectLike, ...] = sum(
            ((r,) if isinstance(r, (_BaseRect, str, np.ndarray)) else tuple(r) for r in rects), ()
        )  # type: ignore
        result: list[_BaseRect] = []
        for r in flat:
            if isinstance(r, _BaseRect):
                result.append(r)
            else:
                rc_cls = _BaseRect.by_dtype(minimal_dtype(r))
                result.append(rc_cls.as_rect(r, copy=False))
        return result

    @classmethod
    def _wrap_ufunc_result(cls, res: Any) -> Any:
        wrapped = super()._wrap_ufunc_result(res)
        if isinstance(wrapped, _BasePointArray) and len(wrapped) == 2:
            target_cls = cls.by_dtype(wrapped._DTYPE)
            out = wrapped.view(target_cls)
            out.flags.writeable = False
            return out
        return wrapped

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            raise TypeError("Rect objects are immutable and do not support in-place operations")

        args = []
        for x in inputs:
            if isinstance(x, np.ndarray):
                args.append(x.view(np.ndarray))
            elif np.isscalar(x) and not isinstance(x, str):
                args.append(x)
            else:
                dt = minimal_dtype(x)
                pt_cls = _BasePoint.by_dtype(dt)
                rc_cls = _BaseRect.by_dtype(dt)
                try:
                    args.append(pt_cls.as_point(x, copy=False).view(np.ndarray))
                except (ValueError, TypeError):
                    try:
                        args.append(rc_cls.as_rect(x, copy=False).view(np.ndarray))
                    except (ValueError, TypeError):
                        args.append(x)

        result = getattr(ufunc, method)(*args, **kwargs)
        if result is NotImplemented:
            return NotImplemented
        if isinstance(result, tuple):
            return tuple(self._wrap_ufunc_result(x) for x in result)
        return self._wrap_ufunc_result(result)

    def __array_finalize__(self, obj):
        if obj is not None and np.shares_memory(self, obj):
            self.flags.writeable = False

    def copy(self, order: Literal["C", "F", "A", "K"] | None = None) -> np.ndarray[Any, np.dtype[DTypeT]]:
        return self.view(np.ndarray).copy(order=order)

    def __setitem__(self, key, value):
        raise TypeError("Rect objects are immutable and do not support item assignment")

    if not TYPE_CHECKING:

        def __getitem__(self, key):
            result = np.ndarray.__getitem__(self, key)
            if isinstance(result, np.ndarray) and result.shape == (2,) and not isinstance(key, tuple):
                return self._POINT_CLS._wrap_as_point(result.copy())
            if isinstance(result, _BaseRect) and result.shape != (2, 2):
                return result.view(np.ndarray)
            return result

        def __and__(self, other):
            return _BaseRect.intersection(self, other)

        def __or__(self, other):
            return _BaseRect.union(self, other)

        def __rand__(self, other):
            return _BaseRect.intersection(self, other)

        def __ror__(self, other):
            return _BaseRect.union(self, other)

    def __eq__(self, other: object) -> bool:
        if np.isscalar(other) and not isinstance(other, str):
            return False
        try:
            other_rect = other if isinstance(other, _BaseRect) else Rect.as_rect(other, copy=False)  # type: ignore
            return bool(np.allclose(self.view(np.ndarray), other_rect.view(np.ndarray)))
        except (ValueError, TypeError):
            return False

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:  # type: ignore
        r = self.view(np.ndarray)
        return hash((float(r[0, 0]), float(r[0, 1]), float(r[1, 0]), float(r[1, 1])))

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.top_left} -> {self.bottom_right})"

    @classmethod
    def bounding_box(cls, points: Any) -> Self:
        pts = cls._ARRAY_CLS.as_array(points, copy=False).view(np.ndarray)
        if len(pts) == 0:
            return cls.empty()
        min_y, min_x = pts.min(axis=0)
        max_y, max_x = pts.max(axis=0)
        as_item = cls._POINT_CLS._as_item
        return cls._from_points(
            cls._POINT_CLS(as_item(min_y), as_item(min_x)),
            cls._POINT_CLS(as_item(max_y), as_item(max_x)),
        )

    @classmethod
    def union(cls, *rects: Iterable[RectLike] | RectLike) -> _BaseRect:
        rects_ = [r for r in cls._flatten_rects(*rects) if not r.is_empty()]
        if len(rects_) == 0:
            return cls.empty()
        if len(rects_) == 1:
            return rects_[0]
        # Promote to Rect if any input is float-based or if called explicitly on Rect
        target_cls = Rect if (cls is Rect or any(isinstance(r, Rect) for r in rects_)) else IntRect
        return target_cls.bounding_box(np.concatenate([r.view(np.ndarray) for r in rects_], axis=0))

    @classmethod
    def intersection(cls, *rects: Iterable[RectLike] | RectLike) -> _BaseRect:
        rects_ = cls._flatten_rects(*rects)
        if len(rects_) == 0 or any(r.is_empty() for r in rects_):
            return cls.empty()
        if len(rects_) == 1:
            return rects_[0]
        target_cls = Rect if (cls is Rect or any(isinstance(r, Rect) for r in rects_)) else IntRect
        points = np.stack([r.view(np.ndarray) for r in rects_], axis=0)
        inter_top_left = np.max(points[:, 0, :], axis=0)
        inter_bottom_right = np.min(points[:, 1, :], axis=0)
        if np.any(inter_top_left > inter_bottom_right):
            return target_cls.empty()
        return target_cls._from_points(
            target_cls._POINT_CLS.as_point(inter_top_left),
            target_cls._POINT_CLS.as_point(inter_bottom_right),
        )

    @property
    def p0(self) -> PointT:
        return self._POINT_CLS._wrap_as_point(self.view(np.ndarray)[0].copy())  # type: ignore

    @property
    def p1(self) -> PointT:
        return self._POINT_CLS._wrap_as_point(self.view(np.ndarray)[1].copy())  # type: ignore

    @property
    def top_left(self) -> PointT:
        r = self.view(np.ndarray)
        return self._POINT_CLS(min(r[0, 0], r[1, 0]), min(r[0, 1], r[1, 1]))  # type: ignore

    @property
    def bottom_right(self) -> PointT:
        r = self.view(np.ndarray)
        return self._POINT_CLS(max(r[0, 0], r[1, 0]), max(r[0, 1], r[1, 1]))  # type: ignore

    @property
    def top_right(self) -> PointT:
        r = self.view(np.ndarray)
        return self._POINT_CLS(min(r[0, 0], r[1, 0]), max(r[0, 1], r[1, 1]))  # type: ignore

    @property
    def bottom_left(self) -> PointT:
        r = self.view(np.ndarray)
        return self._POINT_CLS(max(r[0, 0], r[1, 0]), min(r[0, 1], r[1, 1]))  # type: ignore

    @property
    def sizes(self) -> PointT:
        r = self.view(np.ndarray)
        return self._POINT_CLS(abs(r[1, 0] - r[0, 0]), abs(r[1, 1] - r[0, 1]))  # type: ignore

    @property
    def signed_sizes(self) -> PointT:
        r = self.view(np.ndarray)
        return self._POINT_CLS(r[1, 0] - r[0, 0], r[1, 1] - r[0, 1])  # type: ignore

    @property
    def top(self) -> ItemT:
        r = self.view(np.ndarray)
        return self._POINT_CLS._as_item(min(r[0, 0], r[1, 0]))

    @property
    def left(self) -> ItemT:
        r = self.view(np.ndarray)
        return self._POINT_CLS._as_item(min(r[0, 1], r[1, 1]))

    @property
    def bottom(self) -> ItemT:
        r = self.view(np.ndarray)
        return self._POINT_CLS._as_item(max(r[0, 0], r[1, 0]))

    @property
    def right(self) -> ItemT:
        r = self.view(np.ndarray)
        return self._POINT_CLS._as_item(max(r[0, 1], r[1, 1]))

    @property
    def h(self) -> ItemT:
        r = self.view(np.ndarray)
        return self._POINT_CLS._as_item(abs(r[1, 0] - r[0, 0]))

    @property
    def signed_h(self) -> ItemT:
        r = self.view(np.ndarray)
        return self._POINT_CLS._as_item(r[1, 0] - r[0, 0])

    @property
    def w(self) -> ItemT:
        r = self.view(np.ndarray)
        return self._POINT_CLS._as_item(abs(r[1, 1] - r[0, 1]))

    @property
    def signed_w(self) -> ItemT:
        r = self.view(np.ndarray)
        return self._POINT_CLS._as_item(r[1, 1] - r[0, 1])

    @property
    def area(self) -> ItemT:
        r = self.view(np.ndarray)
        return abs(self._POINT_CLS._as_item((r[1, 0] - r[0, 0]) * (r[1, 1] - r[0, 1])))

    @property
    def tuple(self) -> tuple[tuple[ItemT, ItemT], tuple[ItemT, ItemT]]:
        r = self.view(np.ndarray)
        as_item = self._POINT_CLS._as_item
        return (as_item(r[0, 0]), as_item(r[0, 1])), (as_item(r[1, 0]), as_item(r[1, 1]))

    @property
    def ndarray(self) -> np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[DTypeT]]:
        return self.view(np.ndarray)

    def corners(self) -> PointArrayT:
        r = self.view(np.ndarray)
        p0_top = r[0, 1] < r[1, 1]
        p0_left = r[0, 0] < r[1, 0]
        return self._ARRAY_CLS(
            [
                (r[0, 0] if p0_left else r[1, 0], r[0, 1] if p0_top else r[1, 1]),  # top-left
                (r[0, 0] if p0_left else r[1, 0], r[1, 1] if p0_top else r[0, 1]),  # bottom-left
                (r[1, 0] if p0_left else r[0, 0], r[1, 1] if p0_top else r[0, 1]),  # bottom-right
                (r[1, 0] if p0_left else r[0, 0], r[0, 1] if p0_top else r[1, 1]),  # top-right
            ],
            copy=False,
        )  # type: ignore

    def contains(self, other: PointLike | RectLike) -> bool:
        try:
            pt = other if isinstance(other, _BasePoint) else Point.as_point(other, copy=False)
            p_arr = pt.view(np.ndarray)
            return (self.top <= p_arr[0] <= self.bottom) and (self.left <= p_arr[1] <= self.right)
        except (ValueError, TypeError):
            other_rect = other if isinstance(other, _BaseRect) else Rect.as_rect(other, copy=False)  # type: ignore
            return (
                self.top <= other_rect.top
                and self.left <= other_rect.left
                and self.bottom >= other_rect.bottom
                and self.right >= other_rect.right
            )

    def crop(self, points: PointArrayLike) -> PointArray[Any]:
        pts = PointArray.as_array(points, copy=False)
        rect = self.view(np.ndarray)
        clipped_y = np.clip(pts.y, rect[0, 0], rect[1, 0])
        clipped_x = np.clip(pts.x, rect[0, 1], rect[1, 1])
        return np.stack((clipped_y, clipped_x), axis=-1).view(PointArray)

    @property
    def center(self) -> Point:
        r = self.view(np.ndarray)
        return Point((float(r[0, 0]) + float(r[1, 0])) / 2.0, (float(r[0, 1]) + float(r[1, 1])) / 2.0)

    def as_int(self) -> _BaseRect[np.int64, int, IntPoint, IntPointArray[Literal[2]]]:  # type: ignore
        return IntRect(self.p0.as_int(), self.p1.as_int())

    def as_outer_int(self) -> IntRect:
        r = self.view(np.ndarray)
        h_neg = r[1, 0] < r[0, 0]
        w_neg = r[1, 1] < r[0, 1]
        top_left = self.top_left.floor()
        bottom_right = self.bottom_right.ceil()
        return IntRect(
            self._POINT_CLS(top_left.y if not h_neg else bottom_right.y, top_left.x if not w_neg else bottom_right.x),
            self._POINT_CLS(bottom_right.y if not h_neg else top_left.y, bottom_right.x if not w_neg else top_left.x),
        )

    def as_inner_int(self) -> IntRect:
        r = self.view(np.ndarray)
        h_neg = r[1, 0] < r[0, 0]
        w_neg = r[1, 1] < r[0, 1]
        top_left = self.top_left.ceil()
        bottom_right = self.bottom_right.floor()
        return IntRect(
            self._POINT_CLS(top_left.y if not h_neg else bottom_right.y, top_left.x if not w_neg else bottom_right.x),
            self._POINT_CLS(bottom_right.y if not h_neg else top_left.y, bottom_right.x if not w_neg else top_left.x),
        )

    def as_positive_size(self) -> Self:
        return self._from_points(self.top_left, self.bottom_right)

    def as_float(self) -> _BaseRect[np.float64, float, Point, PointArray[Literal[2]]]:  # type: ignore
        return Rect(self.top_left.as_float(), self.bottom_right.as_float())

    @overload
    def pad(self, pad: ItemT | tuple[ItemT, ItemT], /) -> Self: ...
    @overload
    def pad(self, vertical: ItemT, horizontal: ItemT, /) -> Self: ...
    @overload
    def pad(self, y0: ItemT, x0: ItemT, y1: ItemT, x1: ItemT, /) -> Self: ...
    def pad(self, *pad: ItemT | Iterable[ItemT]) -> Self:
        padding = sum(((p,) if np.isscalar(p) else tuple(p) for p in pad), ())  # type: ignore
        as_item = self._POINT_CLS._as_item
        if len(padding) == 1:
            y0 = x0 = y1 = x1 = as_item(padding[0])
        elif len(padding) == 2:
            y0 = y1 = as_item(padding[0])
            x0 = x1 = as_item(padding[1])
        elif len(padding) == 4:
            y0, x0, y1, x1 = map(as_item, padding)
        else:
            raise ValueError("pad() expects 1, 2, or 4 values for padding")
        rect = self.view(np.ndarray)
        return self._from_points(
            self._POINT_CLS(rect[0, 0] - y0, rect[0, 1] - x0),  # type: ignore[arg-type]
            self._POINT_CLS(rect[1, 0] + y1, rect[1, 1] + x1),  # type: ignore[arg-type]
        )

    ###############################
    ##   Pydantic integration ##
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        from pydantic_core import core_schema

        def validate(value: Any) -> Self:
            if isinstance(value, cls):
                return value
            try:
                return cls.as_rect(value, copy=True)
            except TypeError as e:
                raise ValueError(str(e)) from e

        item_schema = core_schema.float_schema() if np.issubdtype(cls._DTYPE, np.floating) else core_schema.int_schema()
        point_schema = core_schema.tuple_schema([item_schema, item_schema])
        rect_schema = core_schema.tuple_schema([point_schema, point_schema])
        py_schema = core_schema.no_info_plain_validator_function(validate)

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                validate,
                core_schema.union_schema(
                    [
                        rect_schema,
                        core_schema.tuple_schema([item_schema, item_schema, item_schema, item_schema]),
                        point_schema,
                    ]
                ),
            ),
            python_schema=py_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda rect: rect.tuple,
                return_schema=rect_schema,
            ),
        )


if TYPE_CHECKING:
    type _RectInputs = (
        Rect | IntRect | tuple[PointLike, PointLike] | tuple[float, float, float, float] | npt.NDArray[Any]
    )
    type _IntRectInputs = (
        IntRect
        | tuple[IntPointLikeStrict, IntPointLikeStrict]
        | tuple[int, int, int, int]
        | npt.NDArray[np.integer[Any]]
    )


class Rect(
    _GeomOpsMixin[
        "Rect",
        "Rect",
        "Rect",
        "Rect",
        Never,
        Never,
        "PointArrayLike | _RectInputs | float",
        "PointArrayLike | _RectInputs | float",
    ],
    _RectSetOpsMixin["_RectInputs", "_RectInputs", "Rect"],
    _RectIndexMixin[Point, float],
    _BaseRect[np.float64, float, Point, PointArray[Literal[2]]],
):
    """Immutable 2D floating-point rectangle defined by top-left and bottom-right corners."""

    _DTYPE = np.dtype(np.float64)
    _POINT_CLS = Point
    _ARRAY_CLS = PointArray

    @classmethod
    def from_center(cls, center: PointLike, shape: float | PointLike) -> Self:
        c = Point.as_point(center, copy=False)
        halfS = Point.as_point(shape, copy=False) / 2
        return cls._from_points(c - halfS, c + halfS)

    @classmethod
    @functools.cache
    def empty(cls) -> Self:
        return cls._from_points(Point(float("inf"), float("inf")), Point(float("-inf"), float("-inf")))

    def is_empty(self) -> bool:
        if self is Rect.empty():
            return True
        rect = self.view(np.ndarray)
        return not np.isfinite(rect).all() and bool(np.isposinf(rect[1]).all() and np.isneginf(rect[0]).all())


class IntRect(
    _GeomOpsMixin[
        "IntRect",
        "Rect",
        "IntRect",
        "Rect",
        Never,
        Never,
        "IntPointLike | int | _IntRectInputs",
        "PointArrayLike | _RectInputs | float",
    ],
    _RectSetOpsMixin["_IntRectInputs", "_RectInputs", "Rect"],
    _RectIndexMixin[IntPoint, int],
    _BaseRect[np.int64, int, IntPoint, IntPointArray[Literal[2]]],
):
    """Immutable 2D integer rectangle defined by top-left and bottom-right corners."""

    _DTYPE = np.dtype(np.int64)
    _POINT_CLS = IntPoint
    _ARRAY_CLS = IntPointArray

    @classmethod
    def as_rect(cls, s: RectLike | PointLike, *, copy: bool = False) -> Self:
        # Reject non-integer floats during implicit ufunc casting so IntRect + (0.5, 1.5) promotes to Rect
        if np.issubdtype(minimal_dtype(s), np.floating):
            raise TypeError("Non-integer values cannot be cast losslessly to IntRect")
        return super().as_rect(s, copy=copy)

    @classmethod
    def bounding_box(cls, points: PointArrayLike) -> IntRect:
        return Rect.bounding_box(points).as_outer_int()

    @classmethod
    def from_center(cls, center: IntPointLikeStrict, shape: IntPointLikeStrict) -> Self:
        c = IntPoint.as_point(center, copy=False)
        s = IntPoint.as_point(shape, copy=False)
        p0 = c - s // 2
        return cls._from_points(p0, p0 + s)

    @classmethod
    @functools.cache
    def empty(cls) -> Self:
        return cls._from_points(IntPoint(_INT_MAX, _INT_MAX), IntPoint(_INT_MIN, _INT_MIN))

    def is_empty(self) -> bool:
        if self is IntRect.empty():
            return True
        return self.top == _INT_MAX and self.left == _INT_MAX and self.bottom == _INT_MIN and self.right == _INT_MIN

    def as_int(self) -> IntRect:
        return self

    def as_outer_int(self) -> IntRect:
        return self

    def as_inner_int(self) -> IntRect:
        return self

    def to_shape(self) -> tuple[int, int]:
        r = self.view(np.ndarray)
        return (int(abs(r[1, 0] - r[0, 0])), int(abs(r[1, 1] - r[0, 1])))

    @overload
    def grid_indices(
        self, dtype: type[np.integer[Any] | int] = np.int64, exclude_bottom_right: bool = False
    ) -> IntPointMap[Any, Any]: ...
    @overload
    def grid_indices(self, dtype: type[np.float32], exclude_bottom_right: bool = False) -> VectorMap[Any, Any]: ...
    @overload
    def grid_indices(
        self, dtype: type[np.float64 | float], exclude_bottom_right: bool = False
    ) -> PointMap[Any, Any]: ...
    def grid_indices(
        self, dtype: npt.DTypeLike = np.int64, exclude_bottom_right: bool = False
    ) -> _BasePointMap[Any, Any, Any, Any, Any]:
        y1, x1 = self.top_left.floor()
        y2, x2 = self.bottom_right.ceil() - (1 if exclude_bottom_right else 0)
        target_cls = _BasePointMap.by_dtype(dtype)
        yy, xx = np.meshgrid(
            np.arange(y1, y2, dtype=target_cls._DTYPE),
            np.arange(x1, x2, dtype=target_cls._DTYPE),
            indexing="ij",
            copy=False,
            sparse=False,
        )
        return np.stack((yy, xx), axis=-1).view(target_cls)

    def slice(self, exclude_bottom_right: bool = False) -> tuple[slice, slice]:
        offset = 1 if exclude_bottom_right else 0
        return slice(self.top, self.bottom - offset), slice(self.left, self.right - offset)

    def crop_pad_image[T: np.generic](
        self,
        image: npt.NDArray[T],
        origin: IntPointLikeStrict | None = None,
        *,
        channel_last: bool = False,
        copy: bool = True,
    ) -> npt.NDArray[T]:
        if channel_last and image.ndim == 3:
            image = image.transpose(2, 0, 1)

        src_domain = IntRect(image.shape[-2:])
        if origin is not None:
            src_domain = src_domain - IntPoint.as_point(origin)
        src_slice, dst_slice = crop_pad_slices(src_domain, self, prefix_ellipsis=True)

        if copy or not src_domain.contains(self):
            dst_image = np.zeros(image.shape[:-2] + self.to_shape(), dtype=image.dtype)
            dst_image[dst_slice] = image[src_slice]
        else:
            dst_image = image[src_slice]
        if channel_last and dst_image.ndim == 3:
            dst_image = dst_image.transpose(1, 2, 0)
        return dst_image


as_rect = Rect.as_rect
as_int_rect = IntRect.as_rect


def _has_float(obj: Any) -> bool:
    return minimal_dtype(obj) != np.dtype(np.int64)


@overload
def crop_pad_slices(
    src: IntRectLike, dst: IntRectLike, *, prefix_ellipsis: Literal[False] = False
) -> tuple[tuple[slice, slice], tuple[slice, slice]]: ...
@overload
def crop_pad_slices(
    src: IntRectLike, dst: IntRectLike, *, prefix_ellipsis: Literal[True]
) -> tuple[tuple[EllipsisType, slice, slice], tuple[EllipsisType, slice, slice]]: ...
def crop_pad_slices(
    src: IntRectLike, dst: IntRectLike, *, prefix_ellipsis: bool = False
) -> (
    tuple[tuple[slice, slice], tuple[slice, slice]]
    | tuple[tuple[EllipsisType, slice, slice], tuple[EllipsisType, slice, slice]]
):
    """Provide slices to crop/pad 2d matrix from the src domain to the dst domain.

    Parameters
    ----------
    src: Rect | Point | tuple[int, int] | tuple[int, int, int, int]
        The source domain as a rectangle or a shape. If only a shape is provided the origin of the source domain is assumed to be at (0, 0).

    dst: Rect | Point | tuple[int, int] | tuple[int, int, int, int]
        The destination domain as a rectangle or a shape. If only a shape is provided the origin of the destination domain is assumed to be at (0, 0).

    prefix_ellipsis: bool, optional
        If True, add an ellipsis at the beginning of the slices to support images with more than 2 dimensions.


    Returns
        -------
        A tuple of 2 tuples of slices, the first tuple is the slices to crop from the source domain, and the second tuple is the slices to pad into the destination domain.
        The slices are in the format (slice_y, slice_x) and can be used to index into a numpy array.

    Raises
    ------
    TypeError
        If the input is not a Rect or a tuple of 2 integers.
    """
    src_rect, dst_rect = IntRect.as_rect(src), IntRect.as_rect(dst)

    inter = src_rect & dst_rect
    if inter.is_empty():
        slices = (slice(0, 0), slice(0, 0)), (slice(0, 0), slice(0, 0))
    else:
        src_slice = (inter - src_rect.top_left).slice()
        dst_slice = (inter - dst_rect.top_left).slice()
        slices = src_slice, dst_slice

    return ((..., slices[0][0], slices[0][1]), (..., slices[1][0], slices[1][1])) if prefix_ellipsis else slices


type PointLikeStrict = Point | FloatPairTuple | FloatPair
type IntPointLikeStrict = IntPoint | IntPairTuple | IntPair
type AnyPointLikeStrict = IntPointLikeStrict | PointLikeStrict | str
type PointLike = AnyPointLikeStrict | npt.NDArray
type IntPointLike = IntPointLikeStrict | str | npt.NDArray


type PointArrayLikeStrict = (
    PointArray[Any]
    | list[Point]
    | list[IntPoint]
    | list[FloatPairTuple]
    | list[IntPairTuple]
    | list[PointLike]
    | FloatPairArray
)
type PointArrayLike = PointArrayLikeStrict | npt.NDArray | PointLike
type IntPointArrayLikeStrict = (
    IntPointArray[Any] | list[IntPoint] | list[IntPairTuple] | list[IntPointLikeStrict] | IntPairArray
)
type IntPointArrayLike = IntPointArrayLikeStrict | IntPointLikeStrict | npt.NDArray

type PointMapLikeStrict = PointMap[Any, Any] | Sequence[Sequence[PointLike]] | npt.NDArray[np.floating | np.integer]
type PointMapLike = PointMapLikeStrict | PointArrayLike | npt.NDArray
type IntPointMapLikeStrict = IntPointMap[Any, Any] | Sequence[Sequence[IntPointLikeStrict]] | npt.NDArray[np.integer]
type IntPointMapLike = IntPointMapLikeStrict | IntPointArrayLike | npt.NDArray
type VectorMapLikeStrict = VectorMap[Any, Any] | Sequence[Sequence[PointLike]] | npt.NDArray[np.float32]
type VectorMapLike = VectorMapLikeStrict | PointMapLikeStrict | PointArrayLike | npt.NDArray


type IntRectLike = (
    IntRect | tuple[IntPointLikeStrict, IntPointLikeStrict] | tuple[int, int, int, int] | npt.NDArray[np.integer[Any]]
)
type RectLike = Rect | IntRectLike | tuple[PointLike, PointLike] | tuple[float, float, float, float] | npt.NDArray[Any]


ABSENT = Point(np.nan, np.nan)  # Sentinel value for "absent" point (e.g., no intersection)
