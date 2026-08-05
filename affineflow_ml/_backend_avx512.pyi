"""
C++ Core Backend Engine for AffineFlow-ML
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import typing
__all__: list[str] = ['COSINE', 'DISTANCE', 'DistanceMetric', 'EUCLIDEAN', 'KNeighborsClassifier', 'MANHATTAN', 'PCA', 'UNIFORM', 'Weights']
class DistanceMetric:
    """
    Members:
    
      EUCLIDEAN
    
      MANHATTAN
    
      COSINE
    """
    COSINE: typing.ClassVar[DistanceMetric]  # value = <DistanceMetric.COSINE: 2>
    EUCLIDEAN: typing.ClassVar[DistanceMetric]  # value = <DistanceMetric.EUCLIDEAN: 0>
    MANHATTAN: typing.ClassVar[DistanceMetric]  # value = <DistanceMetric.MANHATTAN: 1>
    __members__: typing.ClassVar[dict[str, DistanceMetric]]  # value = {'EUCLIDEAN': <DistanceMetric.EUCLIDEAN: 0>, 'MANHATTAN': <DistanceMetric.MANHATTAN: 1>, 'COSINE': <DistanceMetric.COSINE: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class KNeighborsClassifier:
    """
    K-Nearest Neighbors Classifier
    """
    def __getstate__(self) -> tuple[int, DistanceMetric, Weights, int, typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], list[str], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        ...
    def __init__(self, n_neighbors: typing.SupportsInt | typing.SupportsIndex = 3, metric: DistanceMetric = ..., weights: Weights = ...) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"], y: collections.abc.Sequence[str]) -> None:
        ...
    def predict(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"]) -> list[str]:
        ...
class PCA:
    """
    Principal Component Analysis (PCA)
    """
    def __getstate__(self) -> tuple[float, int, typing.Annotated[numpy.typing.NDArray[numpy.float64], "[1, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[1, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        ...
    def __init__(self, n_components: typing.SupportsFloat | typing.SupportsIndex = 0.95) -> None:
        ...
    def __setstate__(self, arg0: tuple) -> None:
        ...
    def fit(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"], scale: bool = False) -> None:
        ...
    def fit_transform(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"], scale: bool = False) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    def transform(self, X: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
class Weights:
    """
    Members:
    
      UNIFORM
    
      DISTANCE
    """
    DISTANCE: typing.ClassVar[Weights]  # value = <Weights.DISTANCE: 1>
    UNIFORM: typing.ClassVar[Weights]  # value = <Weights.UNIFORM: 0>
    __members__: typing.ClassVar[dict[str, Weights]]  # value = {'UNIFORM': <Weights.UNIFORM: 0>, 'DISTANCE': <Weights.DISTANCE: 1>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
COSINE: DistanceMetric  # value = <DistanceMetric.COSINE: 2>
DISTANCE: Weights  # value = <Weights.DISTANCE: 1>
EUCLIDEAN: DistanceMetric  # value = <DistanceMetric.EUCLIDEAN: 0>
MANHATTAN: DistanceMetric  # value = <DistanceMetric.MANHATTAN: 1>
UNIFORM: Weights  # value = <Weights.UNIFORM: 0>
