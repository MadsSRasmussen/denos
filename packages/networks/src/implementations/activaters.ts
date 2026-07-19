import type { Matrix, MatrixDataType } from "@msrass/matrices";
import { ActivationLayer } from "../models/mod.ts";
import { sigmoid, softmax } from "../utils.ts";

export class ReLUActivation<T extends MatrixDataType>
    extends ActivationLayer<T> {
    protected override activate(input: Matrix<T>): Matrix<T> {
        return input.map((num) => num < 0 ? 0 : num);
    }
    protected override gradient(
        input: Matrix<T>,
        _output: Matrix<T>,
        gradient: Matrix<T>,
    ): Matrix<T> {
        return gradient.hadamard(input.map((num) => num < 0 ? 0 : 1));
    }
}

export class SigmoidActivation<T extends MatrixDataType>
    extends ActivationLayer<T> {
    protected override activate(input: Matrix<T>): Matrix<T> {
        return input.map(sigmoid);
    }
    protected override gradient(
        _input: Matrix<T>,
        output: Matrix<T>,
        gradient: Matrix<T>,
    ): Matrix<T> {
        return gradient.hadamard(output.map((val) => val * (1 - val)));
    }
}

export class SoftmaxActivation<T extends MatrixDataType>
    extends ActivationLayer<T> {
    protected override activate(input: Matrix<T>): Matrix<T> {
        return softmax(input);
    }
    protected override gradient(
        _input: Matrix<T>,
        output: Matrix<T>,
        gradient: Matrix<T>,
    ): Matrix<T> {
        const dotProducts = gradient
            .hadamard(output)
            .reduceRows((acc, val) => acc + val);

        return output.hadamard(
            gradient.subtract(dotProducts),
        );
    }
}
