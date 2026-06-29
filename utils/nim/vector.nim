import std/sequtils
import std/strutils
import std/math

type
  Vector*[T] = ref object
    data: seq[T]

  VectorView*[T] = object
    source*: Vector[T]
    indices: seq[int]

  ContiguousVectorView*[T] = object
    source*: Vector[T]
    ranges*: seq[tuple[lo: int, hi: int]]

