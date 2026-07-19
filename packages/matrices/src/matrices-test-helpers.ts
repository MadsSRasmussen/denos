import { assertEquals } from "@std/assert";
import { Matrix, type MatrixDataType } from "./matrices.ts";

const allocateInt32 = (len: number) => new Int32Array(len);

export function assertMatrixEquals<T extends MatrixDataType>(
    actual: Matrix<T>,
    expected: Matrix<T>,
) {
    assertEquals(actual.toRows(), expected.toRows());
}

export function int32Matrix(matrix: number[][]) {
    return new Matrix(
        matrix.length,
        matrix[0].length,
        allocateInt32,
        new Int32Array(matrix.flat()),
    );
}

export function ascendingInt32Matrix(rows: number, cols: number) {
    const data = allocateInt32(rows * cols);
    for (let i = 0; i < rows * cols; i++) {
        data[i] = i + 1;
    }
    return new Matrix(rows, cols, allocateInt32, data);
}
