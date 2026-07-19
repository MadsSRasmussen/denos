import { assertEquals, assertThrows } from "@std/assert";

import {
    ascendingInt32Matrix as ascM,
    assertMatrixEquals,
    int32Matrix as m,
} from "./matrices-test-helpers.ts";

Deno.test("Matrix access is correct", () => {
    const square = ascM(2, 2);

    assertEquals(square.get(0, 0), 1);
    assertEquals(square.get(0, 1), 2);
    assertEquals(square.get(1, 0), 3);
    assertEquals(square.get(1, 1), 4);

    const column = ascM(1, 4);

    assertEquals(column.get(0, 0), 1);
    assertEquals(column.get(0, 1), 2);
    assertEquals(column.get(0, 2), 3);
    assertEquals(column.get(0, 3), 4);
});

Deno.test("Matrix copy has independent data", async (t) => {
    const cases = [
        {
            name: "regular matrix",
            matrix: ascM(2, 3),
        },
        {
            name: "transposed matrix",
            matrix: ascM(2, 3).transpose(),
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            const copy = test.matrix.copy();

            assertMatrixEquals(copy, test.matrix);
            assertEquals(copy.rowCount, test.matrix.rowCount);
            assertEquals(copy.columnCount, test.matrix.columnCount);
            assertEquals(
                copy.typedArray() === test.matrix.typedArray(),
                false,
            );

            copy.set(0, 0, 99);
            assertEquals(test.matrix.get(0, 0), 1);
        });
    }
});

Deno.test("Matrix multiplication is correct", async (t) => {
    const cases = [
        {
            name: "2x2 time 2x2",
            a: ascM(2, 2),
            b: ascM(2, 2),
            expected: m([[7, 10], [15, 22]]),
        },
        {
            name: "3x2 times 2x2",
            a: ascM(3, 2),
            b: ascM(2, 2),
            expected: m([[7, 10], [15, 22], [23, 34]]),
        },
        {
            name: "2x3T times 2x2",
            a: ascM(2, 3).transpose(),
            b: ascM(2, 2),
            expected: m([[13, 18], [17, 24], [21, 30]]),
        },
        {
            name: "2x2 times 3x2T",
            a: ascM(2, 2),
            b: ascM(3, 2).transpose(),
            expected: m([[5, 11, 17], [11, 25, 39]]),
        },
        {
            name: "3x2T times 2x3T",
            a: ascM(3, 2).transpose(),
            b: ascM(2, 3).transpose(),
            expected: m([[22, 49], [28, 64]]),
        },
        {
            name: "1x4 times 4x1",
            a: m([[1, 2, 0, 3]]),
            b: m([[1], [2], [0], [3]]),
            expected: m([[14]]),
        },
        {
            name: "4x1 times 1x4",
            a: m([[1], [2], [0], [3]]),
            b: m([[1, 2, 0, 3]]),
            expected: m([
                [1, 2, 0, 3],
                [2, 4, 0, 6],
                [0, 0, 0, 0],
                [3, 6, 0, 9],
            ]),
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            assertMatrixEquals(
                test.a.multiply(test.b),
                test.expected,
            );
        });
    }
});

Deno.test("Matrix hadamard operation is correct", async (t) => {
    const cases = [
        {
            name: "2x3 hadamard 2x3",
            a: ascM(2, 3),
            b: ascM(2, 3),
            expected: m([[1, 4, 9], [16, 25, 36]]),
        },
        {
            name: "2x3T hadamard 2x3T",
            a: ascM(2, 3).transpose(),
            b: ascM(2, 3).transpose(),
            expected: m([[1, 16], [4, 25], [9, 36]]),
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            assertMatrixEquals(
                test.a.hadamard(test.b),
                test.expected,
            );
        });
    }
});

Deno.test("Matrix scales correctly", async (t) => {
    const cases = [
        {
            name: "2x3 scales by 2",
            a: ascM(2, 3),
            scalar: 2,
            expected: m([[2, 4, 6], [8, 10, 12]]),
        },
        {
            name: "2x3T scales by 2",
            a: ascM(2, 3).transpose(),
            scalar: 2,
            expected: m([[2, 8], [4, 10], [6, 12]]),
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            assertMatrixEquals(
                test.a.scale(test.scalar),
                test.expected,
            );
        });
    }
});

Deno.test("Matrix creates diagonal matrix", async (t) => {
    const cases = [
        {
            name: "row vector",
            a: ascM(1, 3),
            expected: m([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
        },
        {
            name: "column vector",
            a: ascM(3, 1),
            expected: m([[1, 0, 0], [0, 2, 0], [0, 0, 3]]),
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            assertMatrixEquals(test.a.diagonal(), test.expected);
        });
    }
});

Deno.test("Matrix adds correctly", async (t) => {
    const cases = [
        {
            name: "2x3 plus 2x3",
            a: ascM(2, 3),
            b: ascM(2, 3),
            expected: m([[2, 4, 6], [8, 10, 12]]),
        },
        {
            name: "2x3T plus 2x3T",
            a: ascM(2, 3).transpose(),
            b: ascM(2, 3).transpose(),
            expected: m([[2, 8], [4, 10], [6, 12]]),
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            assertMatrixEquals(
                test.a.add(test.b),
                test.expected,
            );
        });
    }
});

Deno.test("Matrix subtracts correctly", async (t) => {
    const cases = [
        {
            name: "2x3 minus 2x3",
            a: ascM(2, 3),
            b: ascM(2, 3),
            expected: m([[0, 0, 0], [0, 0, 0]]),
        },
        {
            name: "2x3T minus 2x3T",
            a: ascM(2, 3).transpose(),
            b: ascM(2, 3).transpose(),
            expected: m([[0, 0], [0, 0], [0, 0]]),
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            assertMatrixEquals(
                test.a.subtract(test.b),
                test.expected,
            );
        });
    }
});

Deno.test("Matrix operations reject incompatible dimensions", async (t) => {
    const cases = [
        {
            name: "multiplication",
            operation: () => ascM(2, 3).multiply(ascM(2, 3)),
            message: "Cannot multiply",
        },
        {
            name: "hadamard",
            operation: () => ascM(2, 3).hadamard(ascM(3, 2)),
            message: "Cannot broadcast",
        },
        {
            name: "addition",
            operation: () => ascM(2, 3).add(ascM(2, 2)),
            message: "Cannot broadcast",
        },
        {
            name: "subtraction",
            operation: () => ascM(2, 3).subtract(ascM(2, 2)),
            message: "Cannot broadcast",
        },
        {
            name: "diagonal",
            operation: () => ascM(2, 2).diagonal(),
            message: "Cannot compute diagonal",
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            assertThrows(test.operation, RangeError, test.message);
        });
    }
});

Deno.test("Matrix prints correctly", async (t) => {
    const cases = [
        {
            name: "regular matrix",
            matrix: ascM(3, 2),
            expected: `
                [ 1, 2 ]
                [ 3, 4 ]
                [ 5, 6 ]
            `.replaceAll(/^ +/gm, "").trim(),
        },
        {
            name: "transposed matrix",
            matrix: ascM(3, 2).transpose(),
            expected: `
                [ 1, 3, 5 ]
                [ 2, 4, 6 ]
            `.replaceAll(/^ +/gm, "").trim(),
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            assertEquals(test.matrix.toString(), test.expected);
        });
    }
});

Deno.test("Matrix writes correctly into transposed out matrix", async (t) => {
    const cases = [
        {
            name: "add aliasing a transposed operand as out",
            operation: () => {
                const a = ascM(2, 3).transpose(); // logical 3x2
                const b = m([[1, 2], [3, 4], [5, 6]]);
                return a.add(b, a);
            },
            expected: m([[2, 6], [5, 9], [8, 12]]),
        },
        {
            name: "add into an independent transposed out matrix",
            operation: () => {
                const left = ascM(3, 2);
                const right = m([[10, 20], [30, 40], [50, 60]]);
                const out = ascM(2, 3).transpose().copy(); // fresh 3x2, transposed
                return left.add(right, out);
            },
            expected: m([[11, 22], [33, 44], [55, 66]]),
        },
    ];

    for (const test of cases) {
        await t.step(test.name, () => {
            assertMatrixEquals(test.operation(), test.expected);
        });
    }
});

Deno.test("Matrix rejects out matrix with mismatched logical shape", () => {
    // 2x3 and 3x2 have equal length (6) but different logical shape.
    assertThrows(
        () => ascM(2, 3).add(ascM(2, 3), ascM(3, 2)),
        RangeError,
        "requiring shape",
    );
});

Deno.test("Matrix reduceRows and reduceColumns write correctly into out matrix", async (t) => {
    const source = ascM(2, 3); // [[1, 2, 3], [4, 5, 6]]

    await t.step("reduceRows", () => {
        const out = m([[0], [0]]);
        const result = source.reduceRows((acc, val) => acc + val, 0, out);

        assertMatrixEquals(result, m([[6], [15]]));
        assertEquals(result === out, true);
    });

    await t.step("reduceColumns", () => {
        const out = m([[0, 0, 0]]);
        const result = source.reduceColumns((acc, val) => acc + val, 0, out);

        assertMatrixEquals(result, m([[5, 7, 9]]));
        assertEquals(result === out, true);
    });
});
