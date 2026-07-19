import type { Matrix, MatrixDataType } from "@msrass/matrices";
import type { Initializer } from "../models/mod.ts";
import { gaussian, random } from "../utils.ts";

type UniformInitializerConfig = {
    min: number;
    max: number;
};

export class UniformInitializer<T extends MatrixDataType>
    implements Initializer<T> {
    private config: UniformInitializerConfig = {
        min: -1,
        max: 1,
    };

    constructor(config: UniformInitializerConfig) {
        Object.assign(this.config, config);
    }

    public apply(out: Matrix<T>): Matrix<T> {
        return out.map(() => random(this.config.min, this.config.max), out);
    }
}

export class HEInitializer<T extends MatrixDataType> implements Initializer<T> {
    public apply(out: Matrix<T>): Matrix<T> {
        const dev = Math.sqrt(2 / out.rowCount);
        return out.map(() => gaussian() * dev, out);
    }
}

export class XavierInitializer<T extends MatrixDataType>
    implements Initializer<T> {
    public apply(out: Matrix<T>): Matrix<T> {
        const scale = Math.sqrt(2 / out.length);
        return out.map(() => random(-scale, scale), out);
    }
}
