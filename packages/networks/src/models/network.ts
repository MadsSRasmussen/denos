import type { Matrix, MatrixDataType } from "@msrass/matrices";
import type { Initializer, Layer, Parameter } from "./mod.ts";

export class Network<T extends MatrixDataType> implements Layer<T> {
    public layers: Layer<T>[] = [];

    constructor(layers: Layer<T>[]) {
        this.layers = layers;
    }

    public forward(input: Matrix<T>): Matrix<T> {
        for (let i = 0; i < this.layers.length; i++) {
            input = this.layers[i].forward(input);
        }
        return input;
    }

    public backward(gradient: Matrix<T>): Matrix<T> {
        for (let i = this.layers.length - 1; i >= 0; i--) {
            gradient = this.layers[i].backward(gradient);
        }
        return gradient;
    }

    public parameters(): Parameter<T>[] {
        const params = new Set<Parameter<T>>();
        for (const layer of this.layers) {
            for (const param of layer.parameters?.() ?? []) {
                params.add(param);
            }
        }

        return [...params];
    }

    public initialize(strategy?: Initializer<T>) {
        for (const layer of this.layers) {
            if (layer.initialize) layer.initialize(strategy);
        }
    }
}
