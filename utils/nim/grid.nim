import std/sequtils
import std/strutils
import std/math

type
  Grid*[T] = ref object
    rows*, cols*: int
    data*: seq[T]

  GridView*[T] = object
    source*: Grid[T]
    row_indices*: seq[int]
    col_indices*: seq[int]

  ContiguousGridView*[T] = object
    source*: Grid[T]
    row_ranges*: seq[tuple[lo: seq[int], hi: seq[int]]]
    col_ranges*: seq[tuple[lo: seq[int], hi: seq[int]]]
