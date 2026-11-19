from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeGuard, TypeVar

import numpy as np
import numpy.typing as npt

PathLike = str | Path

if TYPE_CHECKING:
    import pandas as pd
    import torch

    Tensor: TypeAlias = torch.Tensor
    TensorOrArray = TypeVar("TensorOrArray", torch.Tensor, npt.NDArray)

type Int1DArray = np.ndarray[tuple[int], np.dtype[np.int_]]
type Int2DArray = np.ndarray[tuple[int, int], np.dtype[np.int_]]
type Int3DArray = np.ndarray[tuple[int, int, int], np.dtype[np.int_]]
type RecursiveIntlist = list[int] | list[RecursiveIntlist]
type IntArrayLike = npt.NDArray[np.integer] | int | RecursiveIntlist
type Int1DArrayLike = npt.NDArray[np.integer] | int | Sequence[int]
type Int2DArrayLike = npt.NDArray[np.integer] | Sequence[int] | Sequence[Sequence[int]]
type Int3DArrayLike = npt.NDArray[np.integer] | Sequence[Sequence[int]] | Sequence[Sequence[Sequence[int]]]

type Bool1DArray = np.ndarray[tuple[int], np.dtype[np.bool_]]
type Bool2DArray = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
type RecursiveBoollist = list[bool] | list[RecursiveBoollist]
type BoolArrayLike = npt.NDArray[np.bool_] | bool | RecursiveBoollist
type Bool1DArrayLike = npt.NDArray[np.bool_] | bool | Sequence[bool]
type Bool2DArrayLike = npt.NDArray[np.bool_] | Sequence[bool] | Sequence[Sequence[bool]]


type RecursiveFloatlist = list[float] | list[RecursiveFloatlist]
type FloatArrayLike = npt.NDArray[np.floating] | float | RecursiveFloatlist
type Float1DArray = np.ndarray[tuple[int], np.dtype[np.float64]]
type Float2DArray = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type Float3DArray = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]
type Float1DArrayLikeStrict = np.ndarray[tuple[int], np.dtype[np.floating]] | list[float] | Int1DArrayLike
type Float1DArrayLike = Float1DArrayLikeStrict | float | npt.NDArray
type Float2DArrayLike = npt.NDArray[np.floating] | Sequence[float] | Sequence[Sequence[float]] | Int2DArrayLike
type Float3DArrayLike = (
    npt.NDArray[np.floating] | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]] | Int3DArrayLike
)


def as_float_1d(arr: Float1DArrayLike) -> Float1DArray:
    arr = np.atleast_1d(arr).astype(np.float64)
    assert arr.ndim == 1, f"Expected a 1D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_float_2d(arr: Float2DArrayLike) -> Float2DArray:
    arr = np.atleast_2d(arr).astype(np.float64)
    assert arr.ndim == 2, f"Expected a 2D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_float_3d(arr: Float3DArrayLike) -> Float3DArray:
    arr = np.atleast_3d(arr).astype(np.float64)
    assert arr.ndim == 3, f"Expected a 3D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_bool_1d(arr: Bool1DArrayLike) -> Bool1DArray:
    arr = np.atleast_1d(arr).astype(np.bool_)
    assert arr.ndim == 1, f"Expected a 1D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_bool_2d(arr: Bool2DArrayLike) -> Bool2DArray:
    arr = np.atleast_2d(arr).astype(np.bool_)
    assert arr.ndim == 2, f"Expected a 2D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_int_1d(arr: Int1DArrayLike) -> Int1DArray:
    arr = np.atleast_1d(arr).astype(np.int_)
    assert arr.ndim == 1, f"Expected a 1D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_int_2d(arr: Int2DArrayLike) -> Int2DArray:
    arr = np.atleast_2d(arr).astype(np.int_)
    assert arr.ndim == 2, f"Expected a 2D array, got array with shape {arr.shape}"
    return arr  # type: ignore


def as_int_3d(arr: Int3DArrayLike) -> Int3DArray:
    arr = np.atleast_3d(arr).astype(np.int_)
    assert arr.ndim == 3, f"Expected a 3D array, got array with shape {arr.shape}"
    return arr  # type: ignore


type FloatPair = np.ndarray[tuple[Literal[2]], np.dtype[np.float64]]
type FloatPairTuple = tuple[float, float]
type FloatPairLike = FloatPairTuple | list[float | int] | npt.NDArray[np.floating | np.integer]

type FloatPairArray = np.ndarray[tuple[int, Literal[2]], np.dtype[np.float64]]
type FloatPairArrayLike = (
    npt.NDArray[np.floating | np.integer]
    | list[float | int]
    | list[list[float | int]]
    | FloatPairTuple
    | Sequence[FloatPairTuple]
    | Sequence[FloatPair]
    | dict[int, float | int]
)
type FloatPairMap = np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.float64]]
type Float32PairMap = np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.float32]]
type FloatPairVolume = np.ndarray[tuple[int, int, int, Literal[2]], np.dtype[np.float64]]
type PointArray = FloatPairArray
type PointArrayLike = FloatPairArrayLike

type IntPair = np.ndarray[tuple[Literal[2]], np.dtype[np.int_]]
type IntPairTuple = tuple[int, int]
type IntPairArray = np.ndarray[tuple[int, Literal[2]], np.dtype[np.int_]]
type IntPairArrayLike = (
    npt.NDArray[np.integer] | list[int] | list[list[int]] | IntPairTuple | Sequence[IntPairTuple] | dict[int, int]
)
type IntPairMap = np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.int_]]
type BoolPairArray = np.ndarray[tuple[int, Literal[2]], np.dtype[np.bool_]]
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


def as_int_pairs(pairs: IntPairArrayLike) -> IntPairArray:
    arr = np.array(list(pairs.items()), dtype=np.int_) if isinstance(pairs, dict) else np.asarray(pairs, dtype=np.int_)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    assert arr.ndim == 2 and arr.shape[1] == 2, "The provided input cannot be converted to a 2D array of shape (N, 2)."
    return arr  # type: ignore


def as_int_pairs_map(pairs: Int3DArrayLike) -> IntPairMap:
    arr = np.asarray(pairs, dtype=np.int_)
    assert arr.ndim == 3 and arr.shape[2] == 2, (
        "The provided input cannot be converted to a 3D array of shape (N, M, 2)."
    )
    return arr  # type: ignore


def as_float_pairs(pairs: FloatPairArrayLike, copy: bool | None = None) -> FloatPairArray:
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


def as_float_pairs_map(pairs: Float3DArrayLike) -> FloatPairMap:
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
