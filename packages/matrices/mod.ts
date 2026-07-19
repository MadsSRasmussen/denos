/**
 * Typed-array-backed two-dimensional matrices and linear algebra operations.
 *
 * Use {@link Matrix} to create matrices with a chosen numeric representation.
 * The provided allocators create matrices backed by common integer and
 * floating-point typed arrays. Matrix arithmetic is available through both
 * instance and static methods. Element-wise operations support broadcasting
 * dimensions of size 1. Results can be written to preallocated matrices to
 * avoid allocating a new Matrix for each operation.
 *
 * @example Create matrices and add a row vector using broadcasting.
 * ```ts
 * import {
 *     allocateFloat64,
 *     Matrix,
 * } from "@msrass/linear/matrices";
 *
 * const matrix = new Matrix(
 *     2,
 *     2,
 *     allocateFloat64,
 *     new Float64Array([1, 2, 3, 4]),
 * );
 * const row = new Matrix(
 *     1,
 *     2,
 *     allocateFloat64,
 *     new Float64Array([10, 20]),
 * );
 * const result = new Matrix(2, 2, allocateFloat64);
 *
 * matrix.add(row, result);
 * result.toRows();
 * // [[11, 22], [13, 24]]
 * ```
 *
 * @module matrices
 */

export * from "./src/matrices.ts";
