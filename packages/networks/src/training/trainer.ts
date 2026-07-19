import type { Matrix, MatrixDataType } from "@msrass/matrices";
import type { LossFunction, Network, Optimizer } from "../models/mod.ts";

export class Trainer<T extends MatrixDataType> {
    private network: Network<T>;
    private loss: LossFunction<T>;
    private optimizer: Optimizer<T>;

    constructor(
        network: Network<T>,
        loss: LossFunction<T>,
        optimizer: Optimizer<T>,
    ) {
        this.network = network;
        this.loss = loss;
        this.optimizer = optimizer;
    }

    public step(input: Matrix<T>, expected: Matrix<T>): number {
        const predicted = this.network.forward(input);
        const { loss, gradient } = this.loss.evaluate(predicted, expected);

        this.network.backward(gradient);

        const parameters = this.network.parameters();
        this.optimizer.step(parameters);

        for (const parameter of parameters) {
            parameter.gradient = undefined;
        }

        return loss;
    }
}
