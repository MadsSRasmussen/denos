import type { Matrix, MatrixDataType } from "@msrass/matrices";

export interface LossFunction<T extends MatrixDataType> {
    evaluate(
        predicted: Matrix<T>,
        expected: Matrix<T>,
    ): {
        loss: number;
        gradient: Matrix<T>;
    };
}
