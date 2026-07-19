import type { MatrixDataType } from "@msrass/matrices";
import type { Parameter } from "./mod.ts";

export interface Optimizer<T extends MatrixDataType> {
    step(parameters: Parameter<T>[]): void;
}
