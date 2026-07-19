import type { Matrix, MatrixDataType } from "@msrass/matrices";
export function sigmoid(input: number): number {
    return 1 / (1 + Math.exp(-input));
}

export function random(min: number, max: number): number {
    return min + Math.random() * (max - min);
}

export function gaussian(): number {
    let u1 = 0;
    let u2 = 0;

    while (u1 === 0) u1 = Math.random();
    while (u2 === 0) u2 = Math.random();

    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

export function softmax<T extends MatrixDataType>(input: Matrix<T>): Matrix<T> {
    const maxima = input.reduceRows(
        (max, val) => Math.max(max, val),
        -Infinity,
    );

    const exponentials = input
        .subtract(maxima)
        .map((val) => Math.exp(val));

    const sums = exponentials.reduceRows((acc, val) => acc + val);

    return exponentials.zip(sums, (exp, sum) => exp / sum);
}
