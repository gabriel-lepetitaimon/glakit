from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, TypeGuard, TypeVar

import numpy as np
import numpy.typing as npt

PathLike = str | Path

if TYPE_CHECKING:
    import pandas as pd
    import torch

    Tensor: TypeAlias = torch.Tensor
    TensorOrArray = TypeVar("TensorOrArray", torch.Tensor, npt.NDArray)

_intlike_dtype = np.dtype[np.integer]
type Int1DArray[N: int] = np.ndarray[tuple[N], np.dtype[np.int_]]
type Int2DArray[M: int, N: int] = np.ndarray[tuple[M, N], np.dtype[np.int_]]
type Int3DArray[M: int, N: int, P: int] = np.ndarray[tuple[M, N, P], np.dtype[np.int_]]
type RecursiveIntlist = list[int] | list[RecursiveIntlist]
type IntArrayLike = npt.NDArray[np.integer] | int | RecursiveIntlist
type Int1DArrayLikeStrict[N: int] = np.ndarray[tuple[N], _intlike_dtype] | Sequence[int]
type Int1DArrayLike[N: int] = np.ndarray[tuple[()], _intlike_dtype] | int | Int1DArrayLikeStrict[N]
type Int2DArrayLike[M: int, N: int] = (
    np.ndarray[tuple[M, N], _intlike_dtype] | Sequence[Sequence[int]] | Int1DArrayLike[N]
)
type Int3DArrayLike[M: int, N: int, P: int] = (
    np.ndarray[tuple[M, N, P], _intlike_dtype] | Sequence[Sequence[Sequence[int]]] | Int2DArrayLike[M, N]
)

_bool_dtype = np.dtype[np.bool_]
type Bool1DArray[N: int] = np.ndarray[tuple[N], _bool_dtype]
type Bool2DArray[M: int, N: int] = np.ndarray[tuple[M, N], _bool_dtype]
type RecursiveBoollist = list[bool] | list[RecursiveBoollist]
type BoolArrayLike = npt.NDArray[np.bool_] | bool | RecursiveBoollist
type Bool1DArrayLike[N: int] = np.ndarray[tuple[N], _bool_dtype] | bool | Sequence[bool]
type Bool2DArrayLike[M: int, N: int] = (
    np.ndarray[tuple[M, N], _bool_dtype] | Sequence[Sequence[bool]] | Bool1DArrayLike[N]
)


_reallike_dtype = np.dtype[np.floating | np.integer]
type RecursiveFloatlist = list[float] | list[RecursiveFloatlist]
type FloatArrayLike = npt.NDArray[np.floating] | float | RecursiveFloatlist
type Float1DArray[N: int] = np.ndarray[tuple[N], np.dtype[np.float64]]
type Float2DArray[M: int, N: int] = np.ndarray[tuple[M, N], np.dtype[np.float64]]
type Float3DArray[M: int, N: int, P: int] = np.ndarray[tuple[M, N, P], np.dtype[np.float64]]
type Float1DArrayLikeStrict[N: int] = np.ndarray[tuple[N], _reallike_dtype] | Sequence[float]
type Float1DArrayLike[N: int] = np.ndarray[tuple[()], _reallike_dtype] | float | Float1DArrayLikeStrict[N]
type Float2DArrayLike[M: int, N: int] = (
    np.ndarray[tuple[M, N], _reallike_dtype] | Sequence[Sequence[float]] | Float1DArrayLike[N]
)
type Float3DArrayLike[M: int, N: int, P: int] = (
    np.ndarray[tuple[M, N, P], _reallike_dtype] | Sequence[Sequence[Sequence[float]]] | Float2DArrayLike[M, N]
)


def as_float_1d[N: int](arr: Float1DArrayLike[N]) -> Float1DArray[N]:
    arr = np.atleast_1d(arr).astype(np.float64)
    assert arr.ndim == 1, f"Expected a 1D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_float_2d[M: int, N: int](arr: Float2DArrayLike[M, N]) -> Float2DArray[M, N]:
    arr = np.atleast_2d(arr).astype(np.float64)
    assert arr.ndim == 2, f"Expected a 2D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_float_3d[M: int, N: int, P: int](arr: Float3DArrayLike[M, N, P]) -> Float3DArray[M, N, P]:
    arr = np.atleast_3d(arr).astype(np.float64)
    assert arr.ndim == 3, f"Expected a 3D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_bool_1d[N: int](arr: Bool1DArrayLike[N]) -> Bool1DArray[N]:
    arr = np.atleast_1d(arr).astype(np.bool_)
    assert arr.ndim == 1, f"Expected a 1D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_bool_2d[M: int, N: int](arr: Bool2DArrayLike[M, N]) -> Bool2DArray[M, N]:
    arr = np.atleast_2d(arr).astype(np.bool_)
    assert arr.ndim == 2, f"Expected a 2D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_int_1d[N: int](arr: Int1DArrayLike[N]) -> Int1DArray[N]:
    arr = np.atleast_1d(arr).astype(np.int_)
    assert arr.ndim == 1, f"Expected a 1D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_int_2d[M: int, N: int](arr: Int2DArrayLike[M, N]) -> Int2DArray[M, N]:
    arr = np.atleast_2d(arr).astype(np.int_)
    assert arr.ndim == 2, f"Expected a 2D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_int_3d[M: int, N: int, P: int](arr: Int3DArrayLike[M, N, P]) -> Int3DArray[M, N, P]:
    arr = np.atleast_3d(arr).astype(np.int_)
    assert arr.ndim == 3, f"Expected a 3D array, got array with shape {arr.shape}"
    return arr  # type: ignore


type FloatPair = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
type FloatPairTuple = tuple[float, float]
type FloatPairLike = FloatPairTuple | list[float | int] | npt.NDArray[np.floating | np.integer]

type FloatPairArray[N: int] = np.ndarray[tuple[N, Literal[2]], np.dtype[np.float64]]
type FloatPairArrayLike = (
    npt.NDArray[np.floating | np.integer]
    | list[float | int]
    | list[list[float | int]]
    | FloatPairTuple
    | Sequence[FloatPairTuple]
    | Sequence[FloatPair]
    | dict[int, float | int]
)
type FloatPairMap[M: int, N: int] = np.ndarray[tuple[M, N, Literal[2]], np.dtype[np.float64]]
type Float32PairMap[M: int, N: int] = np.ndarray[tuple[M, N, Literal[2]], np.dtype[np.float32]]
type FloatPairVolume[M: int, N: int, P: int] = np.ndarray[tuple[M, N, P, Literal[2]], np.dtype[np.float64]]
type PointArray[N: int] = FloatPairArray[N]
type PointArrayLike = FloatPairArrayLike

type IntPair = np.ndarray[tuple[Literal[2]], np.dtype[np.int_]]
type IntPairTuple = tuple[int, int]
type IntPairArray[N: int] = np.ndarray[tuple[N, Literal[2]], np.dtype[np.int_]]
type IntPairArrayLike = (
    npt.NDArray[np.integer] | list[int] | list[list[int]] | IntPairTuple | Sequence[IntPairTuple] | dict[int, int]
)
type IntPairMap[M: int, N: int] = np.ndarray[tuple[M, N, Literal[2]], np.dtype[np.int_]]
type BoolPairArray[N: int] = np.ndarray[tuple[N, Literal[2]], np.dtype[np.bool_]]
type BoolPairArrayLike = (
    npt.NDArray[np.bool_] | list[bool] | list[list[bool]] | tuple[bool, bool] | Sequence[tuple[bool, bool]]
)

type Index = int
type Indices = npt.NDArray[np.int_]
type IndicesLike = Index | npt.NDArray[np.integer] | list[int] | pd.Series


def as_float_pair(pair: FloatPairLike) -> FloatPair:
    arr = np.asarray(pair, dtype=np.float64)
    assert arr.shape == (2,), "The provided input cannot be converted to a 2D point."
    return arr  # type: ignore


def as_int_pairs(pairs: IntPairArrayLike) -> IntPairArray[Any]:
    arr = np.array(list(pairs.items()), dtype=np.int_) if isinstance(pairs, dict) else np.asarray(pairs, dtype=np.int_)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    assert arr.ndim == 2 and arr.shape[1] == 2, "The provided input cannot be converted to a 2D array of shape (N, 2)."
    return arr  # type: ignore


def as_int_pairs_map(pairs: Int3DArrayLike) -> IntPairMap[Any, Any]:
    arr = np.asarray(pairs, dtype=np.int_)
    assert arr.ndim == 3 and arr.shape[2] == 2, (
        "The provided input cannot be converted to a 3D array of shape (N, M, 2)."
    )
    return arr  # type: ignore


def as_float_pairs(pairs: FloatPairArrayLike, copy: bool | None = None) -> FloatPairArray[Any]:
    arr = (
        np.array(list(pairs.items()), dtype=np.float64)
        if isinstance(pairs, dict)
        else np.asarray(pairs, dtype=np.float64, copy=copy)
    )
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    assert arr.ndim == 2 and arr.shape[1] == 2, "The provided input cannot be converted to a 2D array of shape (N, 2)."
    return arr  # type: ignore


as_points = as_float_pairs


def as_float_pairs_map(pairs: Float3DArrayLike) -> FloatPairMap[Any, Any]:
    arr = np.asarray(pairs, dtype=np.float64)
    assert arr.ndim == 3 and arr.shape[2] == 2, (
        "The provided input cannot be converted to a 3D array of shape (N, M, 2)."
    )
    return arr  # type: ignore


def is_torch_tensor(x) -> TypeGuard[torch.Tensor]:
    """Check if the input is a torch.Tensor.

    If the type of the input not named "Tensor", it will return False without importing torch.
    """
    if type(x).__qualname__ == "Tensor":
        try:
            import torch
        except ImportError:
            return False
        return isinstance(x, torch.Tensor)
    return False
