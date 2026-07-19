import type { Matrix, MatrixDataType } from "@msrass/matrices";
import type { LossFunction } from "../models/mod.ts";

export class CategoricalCrossEntropy<T extends MatrixDataType>
    implements LossFunction<T> {
    public evaluate(
        predicted: Matrix<T>,
        expected: Matrix<T>,
    ): { loss: number; gradient: Matrix<T> } {
        const rowCount = predicted.rowCount;
        const probabilities = predicted.map(
            (value) => Math.max(value, 1e-7),
        );

        const loss = probabilities
            .map((value) => Math.log(value))
            .hadamard(expected)
            .scale(-1 / rowCount)
            .reduce((sum, value) => sum + value);

        const gradient = probabilities.zip(
            expected,
            (probability, target) => -target / (probability * rowCount),
        );

        return { loss, gradient };
    }
}

export class MeanSquaredError<T extends MatrixDataType>
    implements LossFunction<T> {
    public evaluate(
        predicted: Matrix<T>,
        expected: Matrix<T>,
    ): { loss: number; gradient: Matrix<T> } {
        const loss = predicted.subtract(expected)
            .map((val) => 0.5 * val * val)
            .scale(1 / predicted.length)
            .reduce((acc, val) => acc + val);

        const gradient = predicted.subtract(expected).scale(
            1 / predicted.length,
        );

        return { loss, gradient };
    }
}
