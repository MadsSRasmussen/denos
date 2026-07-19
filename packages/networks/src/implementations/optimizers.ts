import type { MatrixDataType } from "@msrass/matrices";
import type { Optimizer, Parameter } from "../models/mod.ts";

type GradientDescentConfig = {
    rate: number;
};

export class GradientDescent<T extends MatrixDataType> implements Optimizer<T> {
    private config: GradientDescentConfig = {
        rate: 0.01,
    };

    constructor(config: Partial<GradientDescentConfig> = {}) {
        Object.assign(this.config, config);
    }

    public step(parameters: Parameter<T>[]): void {
        for (const parameter of parameters) {
            if (!parameter.gradient) continue;

            parameter.value.zip(
                parameter.gradient,
                (value, gradient) => value - gradient * this.config.rate,
                parameter.value,
            );
        }
    }
}
