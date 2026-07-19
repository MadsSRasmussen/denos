import {
    allocateFloat32,
    type Allocator,
    Matrix,
    type MatrixDataType,
} from "@msrass/matrices";
import type { Initializer } from "./mod.ts";

import { UniformInitializer } from "../implementations/initializers.ts";

export interface Parameter<T extends MatrixDataType> {
    value: Matrix<T>;
    gradient?: Matrix<T>;
}

export interface Layer<T extends MatrixDataType> {
    forward(input: Matrix<T>): Matrix<T>;
    backward(gradiant: Matrix<T>): Matrix<T>;
    parameters?(): Parameter<T>[];
    initialize?(strategy?: Initializer<T>): void;
}

type DenseLayerConfig<T extends MatrixDataType> = {
    initializer: Initializer<T>;
    allocator: Allocator<T>;
};

export class DenseLayer<T extends MatrixDataType = Float32Array>
    implements Layer<T> {
    public weights: Parameter<T>;
    public biases: Parameter<T>;
    private initializer: Initializer<T>;

    private input?: Matrix<T>;

    constructor(
        input: number,
        output: number,
        config?: Partial<DenseLayerConfig<Float32Array>>,
    );
    constructor(input: number, output: number, config: DenseLayerConfig<T>);
    constructor(
        input: number,
        output: number,
        config: Partial<DenseLayerConfig<T>> = {},
    ) {
        const allocator = (config.allocator ?? allocateFloat32) as Allocator<T>;

        this.weights = { value: new Matrix(input, output, allocator) };
        this.biases = { value: new Matrix(1, output, allocator) };
        this.initializer = (config.initializer ??
            new UniformInitializer<T>({ min: -1, max: 1 })) as Initializer<T>;
    }

    forward(input: Matrix<T>): Matrix<T> {
        this.input = input;
        return input.multiply(this.weights.value).add(this.biases.value);
    }

    backward(gradient: Matrix<T>): Matrix<T> {
        if (!this.input) throw new Error();
        const inputGradiant = gradient.multiply(this.weights.value.transpose());
        this.weights.gradient = this.input.transpose().multiply(gradient);
        this.biases.gradient = gradient.reduceColumns((acc, val) => acc + val);
        return inputGradiant;
    }

    parameters(): Parameter<T>[] {
        return [this.weights, this.biases];
    }

    initialize(strategy?: Initializer<T>) {
        const initializer = strategy ?? this.initializer;
        this.weights.value = initializer.apply(this.weights.value);
        this.biases.value = this.biases.value.map(() => 0);

        this.weights.gradient = undefined;
        this.biases.gradient = undefined;
    }
}

export abstract class ActivationLayer<T extends MatrixDataType>
    implements Layer<T> {
    private input?: Matrix<T>;
    private output?: Matrix<T>;

    protected abstract activate(input: Matrix<T>): Matrix<T>;
    protected abstract gradient(
        input: Matrix<T>,
        output: Matrix<T>,
        gradient: Matrix<T>,
    ): Matrix<T>;

    forward(input: Matrix<T>): Matrix<T> {
        this.input = input;
        this.output = this.activate(input);
        return this.output;
    }

    backward(gradient: Matrix<T>): Matrix<T> {
        if (!this.input || !this.output) throw new Error();
        return this.gradient(this.input, this.output, gradient);
    }
}
