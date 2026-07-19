import type { Matrix, MatrixDataType } from "@msrass/matrices";

export interface Initializer<T extends MatrixDataType> {
    apply(out: Matrix<T>): Matrix<T>;
}
