/**
 * The underlying data type storing the matrix data.
 */
export type MatrixDataType =
    | Int8Array
    | Int16Array
    | Int32Array
    | Float16Array
    | Float32Array
    | Float64Array;

/**
 * Function allocating a certain type of matrix data.
 */
export type Allocator<T extends MatrixDataType> = (len: number) => T;

/**
 * Function wrapper for allocating Int32Arrays
 *
 * @param len The length of the typed array allocation.
 */
export function allocateInt32(len: number): Int32Array {
    return new Int32Array(len);
}

/**
 * Function wrapper for allocating Float32Arrays
 *
 * @param len The length of the typed array allocation.
 */
export function allocateFloat32(len: number): Float32Array {
    return new Float32Array(len);
}

/**
 * Function wrapper for allocating Float32Arrays
 *
 * @param len The length of the typed array allocation.
 */
export function allocateFloat64(len: number): Float64Array {
    return new Float64Array(len);
}

/**
 * Function wrapper for allocating Float32Arrays
 *
 * @param len The length of the typed array allocation.
 */
export function allocateFloat16(len: number): Float16Array {
    return new Float16Array(len);
}

/**
 * Matrix representation of a 2D matrix of numbers.
 */
export class Matrix<T extends MatrixDataType> {
    private static ensure<T extends MatrixDataType>(
        out: Matrix<T> | undefined,
        rows: number,
        columns: number,
        allocator: Allocator<T>,
    ) {
        if (!out) {
            return new Matrix(rows, columns, allocator);
        } else if (out.rowCount !== rows || out.columnCount !== columns) {
            throw new RangeError(
                `Cannot use ${out.rowCount}x${out.columnCount} out matrix for result requiring shape ${rows}x${columns}.`,
            );
        }
        return out;
    }

    private static overlaps(
        a: MatrixDataType,
        b: MatrixDataType,
    ) {
        return a.buffer === b.buffer;
    }

    /**
     * Map a Matrix by applying the specified transform operation to each
     * element of the Matrix.
     *
     * @param a The Matrix to map
     * @param transform The transform function applied to each element
     * @param out The optional Matrix to write the result to. It must have the
     * same shape as the Matrix being mapped. When provided, it is overwritten
     * and returned instead of allocating a Matrix
     */
    public static map<T extends MatrixDataType>(
        a: Matrix<T>,
        transform: (val: number) => number,
        out?: Matrix<T>,
    ): Matrix<T> {
        out = Matrix.ensure(
            out,
            a.rowCount,
            a.columnCount,
            a.allocator,
        );
        out = Matrix.ensure(out, a.rowCount, a.columnCount, a.allocator);
        for (let i = 0; i < a.rowCount; i++) {
            for (let j = 0; j < a.columnCount; j++) {
                out.set(i, j, transform(a.get(i, j)));
            }
        }

        return out;
    }

    /**
     * Reduce a Matrix to a number applying the specified accumulator
     * function on each element of the Matrix.
     *
     * @param a The Matrix to reduce
     * @param transform The accumulator function applied to each element
     * @param initial The initial value of the number accumulated
     */
    public static reduce<T extends MatrixDataType>(
        a: Matrix<T>,
        transform: (acc: number, val: number) => number,
        initial: number = 0,
    ): number {
        let acc = initial;
        for (let i = 0; i < a.length; i++) {
            acc = transform(acc, a.data[i]);
        }
        return acc;
    }

    /**
     * Reduce each row of a Matrix to a column vector.
     *
     * Each row is reduced independently from left to right, starting with the
     * specified initial value.
     *
     * @param a The Matrix whose rows to reduce
     * @param transform The accumulator function applied to each element
     * @param initial The initial value accumulated for each row
     * @param out The optional Matrix to write the result to. It must have the
     * same number of rows as the input and one column. When provided, it is
     * overwritten and returned instead of allocating a Matrix
     */
    public static reduceRows<T extends MatrixDataType>(
        a: Matrix<T>,
        transform: (acc: number, val: number) => number,
        initial: number = 0,
        out?: Matrix<T>,
    ): Matrix<T> {
        out = Matrix.ensure(out, a.rowCount, 1, a.allocator);

        let acc: number;
        for (let i = 0; i < a.rowCount; i++) {
            acc = initial;
            for (let j = 0; j < a.columnCount; j++) {
                acc = transform(acc, a.get(i, j));
            }
            out.set(i, 0, acc);
        }

        return out;
    }

    /**
     * Reduce each column of a Matrix to a row vector.
     *
     * Each column is reduced independently from top to bottom, starting with
     * the specified initial value.
     *
     * @param a The Matrix whose columns to reduce
     * @param transform The accumulator function applied to each element
     * @param initial The initial value accumulated for each column
     * @param out The optional Matrix to write the result to. It must have one
     * row and the same number of columns as the input. When provided, it is
     * overwritten and returned instead of allocating a Matrix
     */
    public static reduceColumns<T extends MatrixDataType>(
        a: Matrix<T>,
        transform: (acc: number, val: number) => number,
        initial: number = 0,
        out?: Matrix<T>,
    ): Matrix<T> {
        out = Matrix.ensure(out, 1, a.columnCount, a.allocator);

        let acc: number;
        for (let j = 0; j < a.columnCount; j++) {
            acc = initial;
            for (let i = 0; i < a.rowCount; i++) {
                acc = transform(acc, a.get(i, j));
            }
            out.set(0, j, acc);
        }

        return out;
    }

    /**
     * Determine whether two Matrices can be broadcast together.
     *
     * For each dimension, the sizes must be equal or one of them must be 1.
     *
     * @param a The first Matrix
     * @param b The second Matrix
     */
    public static broadcastable<T extends MatrixDataType>(
        a: Matrix<T>,
        b: Matrix<T>,
    ): boolean {
        const rows = a.rowCount === b.rowCount || a.rowCount === 1 ||
            b.rowCount === 1;
        const cols = a.columnCount === b.columnCount || a.columnCount === 1 ||
            b.columnCount === 1;
        return rows && cols;
    }

    /**
     * Combine two Matrices by applying the specified transform to pairs of
     * elements.
     *
     * The Matrices must be broadcastable. For each dimension, the sizes must
     * be equal or one of them must be 1. The resulting Matrix has the maximum
     * row and column counts of the two Matrices.
     *
     * @param a The first Matrix
     * @param b The second Matrix
     * @param transform The transform function applied to each pair of elements
     * @param out The optional Matrix to write the result to. It must have the
     * resulting broadcast shape. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    public static zip<T extends MatrixDataType>(
        a: Matrix<T>,
        b: Matrix<T>,
        transform: (x: number, y: number) => number,
        out?: Matrix<T>,
    ): Matrix<T> {
        if (!Matrix.broadcastable(a, b)) {
            throw new RangeError(
                `Cannot broadcast ${a.rowCount}x${a.columnCount} with ${b.rowCount}x${b.columnCount} matricies.`,
            );
        }
        const rows = Math.max(a.rowCount, b.rowCount);
        const cols = Math.max(a.columnCount, b.columnCount);

        out = Matrix.ensure(out, rows, cols, a.allocator);
        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                out.set(
                    i,
                    j,
                    transform(
                        a.get(i % a.rowCount, j % a.columnCount),
                        b.get(i % b.rowCount, j % b.columnCount),
                    ),
                );
            }
        }

        return out;
    }

    /**
     * Scale a Matrix by a scalar value.
     *
     * @param a The Matrix to scale
     * @param scalar The number to scale values by
     * @param out The optional Matrix to write the result to. It must have the
     * same shape as the Matrix being scaled. When provided, it is overwritten
     * and returned instead of allocating a Matrix
     */
    public static scale<T extends MatrixDataType>(
        a: Matrix<T>,
        scalar: number,
        out?: Matrix<T>,
    ): Matrix<T> {
        return Matrix.map(a, (x) => x * scalar, out);
    }

    /**
     * Addition of one Matrix with another.
     *
     * The Matrices must be broadcastable. For each dimension, the sizes must
     * be equal or one of them must be 1.
     *
     * @param a The Matrix to add to
     * @param b The Matrix to add
     * @param out The optional Matrix to write the result to. It must have the
     * resulting broadcast shape. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    public static add<T extends MatrixDataType>(
        a: Matrix<T>,
        b: Matrix<T>,
        out?: Matrix<T>,
    ): Matrix<T> {
        return Matrix.zip(a, b, (x, y) => x + y, out);
    }

    /**
     * Subtraction of one Matrix from another.
     *
     * The Matrices must be broadcastable. For each dimension, the sizes must
     * be equal or one of them must be 1.
     *
     * @param a The Matrix to subtract from
     * @param b The Matrix to subtract
     * @param out The optional Matrix to write the result to. It must have the
     * resulting broadcast shape. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    public static subtract<T extends MatrixDataType>(
        a: Matrix<T>,
        b: Matrix<T>,
        out?: Matrix<T>,
    ): Matrix<T> {
        return Matrix.zip(a, b, (x, y) => x - y, out);
    }

    /**
     * Hadamard product of two Matrices.
     *
     * The Matrices must be broadcastable. For each dimension, the sizes must
     * be equal or one of them must be 1.
     *
     * @param a The Matrix with the multiplicand values
     * @param b The Matrix with the multiplier values
     * @param out The optional Matrix to write the result to. It must have the
     * resulting broadcast shape. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    public static hadamard<T extends MatrixDataType>(
        a: Matrix<T>,
        b: Matrix<T>,
        out?: Matrix<T>,
    ): Matrix<T> {
        return Matrix.zip(a, b, (x, y) => x * y, out);
    }

    /**
     * Multiply a left factor Matrix with a right factor Matrix.
     *
     * The number of columns in the left Matrix must equal the number of rows
     * in the right Matrix.
     *
     * @param a The left factor Matrix in the multiplication
     * @param b The right factor Matrix in the multiplication
     * @param out The optional Matrix to write the result to. It must have the
     * left Matrix row count and right Matrix column count, and must not share
     * underlying data with either factor. When provided, it is overwritten and
     * returned instead of allocating a Matrix
     */
    public static multiply<T extends MatrixDataType>(
        a: Matrix<T>,
        b: Matrix<T>,
        out?: Matrix<T>,
    ): Matrix<T> {
        if (a.columnCount !== b.rowCount) {
            throw new RangeError(
                `Cannot multiply ${a.rowCount}x${a.columnCount} by ${b.rowCount}x${b.columnCount} matricies.`,
            );
        }

        if (
            out &&
            (Matrix.overlaps(out.data, a.data) ||
                Matrix.overlaps(out.data, b.data))
        ) {
            throw new RangeError(
                `Cannot write to output matrix overlapping input matrices.`,
            );
        }

        out = Matrix.ensure(
            out,
            a.rowCount,
            b.columnCount,
            a.allocator,
        );

        let acc;
        for (let i = 0; i < a.rowCount; i++) {
            for (let j = 0; j < b.columnCount; j++) {
                acc = 0;
                for (let k = 0; k < b.rowCount; k++) {
                    acc += a.get(i, k) * b.get(k, j);
                }
                out.data[i * b.columnCount + j] = acc;
            }
        }

        return out;
    }

    /**
     * Compute a diagonal Matrix from a row- or column vector.
     *
     * The Matrix must be either a row- or a column vector.
     *
     * @param a The row- or column vector
     */
    public static diagonal<T extends MatrixDataType>(a: Matrix<T>): Matrix<T> {
        if (!(a.rowCount === 1 || a.columnCount === 1)) {
            throw new RangeError(
                `Cannot compute diagonal matrix from ${a.rowCount}x${a.columnCount} matrix`,
            );
        }

        const out = new Matrix(a.length, a.length, a.allocator);
        for (let i = 0; i < a.length; i++) {
            out.set(i, i, a.data[i]);
        }

        return out;
    }

    /**
     * Create a deep copy of a Matrix.
     *
     * @param a The Matrix to copy
     */
    public static copy<T extends MatrixDataType>(a: Matrix<T>): Matrix<T> {
        const data = a.allocator(a.length);
        data.set(a.data);

        const result = new Matrix(
            a._rowCount,
            a._columnCount,
            a.allocator,
            data,
        );
        result.transposed = a.transposed;
        return result;
    }

    /**
     * Return a transposed view of a Matrix.
     *
     * The operation does not mutate the original Matrix. By default, the
     * returned view shares its underlying data with the original Matrix.
     *
     * @param a The Matrix to transpose
     * @param copy Whether to copy the underlying data
     */
    public static transpose<T extends MatrixDataType>(
        a: Matrix<T>,
        copy: boolean = false,
    ): Matrix<T> {
        const result = copy
            ? Matrix.copy(a)
            : new Matrix(a._rowCount, a._columnCount, a.allocator, a.data);
        result.transposed = !a.transposed;
        return result;
    }

    /** The total length of the underlying data array */
    public length: number;

    private allocator: Allocator<T>;

    /** The typed array storing the Matrix values. */
    protected data: T;

    private _rowCount: number;
    private _columnCount: number;

    private transposed: boolean = false;

    /**
     * Create a Matrix with the specified dimensions and data representation.
     *
     * When data is provided, its length must equal the product of the row and
     * column counts. The Matrix uses the provided typed array directly.
     *
     * @param rowCount The number of rows in the Matrix
     * @param columnCount The number of columns in the Matrix
     * @param allocator The function used to allocate typed arrays
     * @param data The optional typed array containing the Matrix values
     */
    constructor(
        rowCount: number,
        columnCount: number,
        allocator: Allocator<T>,
        data?: T,
    ) {
        if (data && data.length !== rowCount * columnCount) {
            throw new Error(
                `Cannot instantiate Matrix with data length ${data.length} with row count ${rowCount} and column count ${columnCount}`,
            );
        }

        this._rowCount = rowCount;
        this._columnCount = columnCount;
        this.allocator = allocator;

        this.length = rowCount * columnCount;
        if (data) {
            this.data = data;
        } else {
            this.data = this.allocator(this.length);
        }
    }

    /** The number of rows in the Matrix. */
    get rowCount(): number {
        if (this.transposed) {
            return this._columnCount;
        } else {
            return this._rowCount;
        }
    }

    /** The number of columns in the Matrix. */
    get columnCount(): number {
        if (this.transposed) {
            return this._rowCount;
        } else {
            return this._columnCount;
        }
    }

    /**
     * Set a value at a specific row- and column coordinate in the Matrix.
     *
     * @param row The row index to set
     * @param col The column index to set
     * @param value The value to set
     */
    set(row: number, col: number, value: number): void {
        if (this.transposed) {
            this.data[col * this._columnCount + row] = value;
        } else {
            this.data[row * this._columnCount + col] = value;
        }
    }

    /**
     * Get a value at a specific row- and column coordinate in the Matrix.
     *
     * @param row The row index to get
     * @param col The column index to get
     */
    get(row: number, col: number): number {
        if (this.transposed) {
            const idk = this.data[col * this._columnCount + row];
            return idk;
        } else {
            return this.data[row * this._columnCount + col];
        }
    }

    /** Creates a deep copy of the Matrix. */
    copy(): Matrix<T> {
        return Matrix.copy(this);
    }

    /**
     * The underlying data typed array.
     *
     * This exposes the underlying mutable internal data representation in storage order.
     * This might differ from row order if the Matrix is transposed.
     */
    typedArray(): T {
        return this.data;
    }

    /**
     * Multiply this Matrix as the left factor in a multiplication with another right factor Matrix.
     *
     * The number of columns in this left Matrix must equal the number of rows in that right Matrix.
     *
     * @param that The right factor Matrix in the multiplication.
     * @param out The optional Matrix to write the result to. It must have this
     * Matrix's row count and the right Matrix's column count, and must not share
     * underlying data with either factor. When provided, it is overwritten and
     * returned instead of allocating a Matrix
     */
    multiply(that: Matrix<T>, out?: Matrix<T>): Matrix<T> {
        return Matrix.multiply(this, that, out);
    }

    /**
     * Hadamard product of this Matrix with elements acting as the multiplicand
     * with another Matrix with multiplier elements.
     *
     * The Matrices must be broadcastable. For each dimension, the sizes must
     * be equal or one of them must be 1.
     *
     * @param that The Matrix with the multiplier values
     * @param out The optional Matrix to write the result to. It must have the
     * resulting broadcast shape. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    hadamard(that: Matrix<T>, out?: Matrix<T>): Matrix<T> {
        return Matrix.hadamard(this, that, out);
    }

    /**
     * Addition of this Matrix with another.
     *
     * The Matrices must be broadcastable. For each dimension, the sizes must
     * be equal or one of them must be 1.
     *
     * @param that The Matrix to add
     * @param out The optional Matrix to write the result to. It must have the
     * resulting broadcast shape. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    add(that: Matrix<T>, out?: Matrix<T>): Matrix<T> {
        return Matrix.add(this, that, out);
    }

    /**
     * Subtraction of this Matrix with another.
     *
     * The Matrices must be broadcastable. For each dimension, the sizes must
     * be equal or one of them must be 1.
     *
     * @param that The Matrix to subtract
     * @param out The optional Matrix to write the result to. It must have the
     * resulting broadcast shape. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    subtract(that: Matrix<T>, out?: Matrix<T>): Matrix<T> {
        return Matrix.subtract(this, that, out);
    }

    /**
     * Scale this Matrix by a scalar value.
     *
     * @param scalar The number to scale values by
     * @param out The optional Matrix to write the result to. It must have the
     * same shape as this Matrix. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    scale(scalar: number, out?: Matrix<T>): Matrix<T> {
        return Matrix.scale(this, scalar, out);
    }

    /**
     * Map this Matrix by applying the specified transform operation to each
     * element of the Matrix.
     *
     * @param transform The transform function applied to each element
     * @param out The optional Matrix to write the result to. It must have the
     * same shape as this Matrix. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    map(transform: (val: number) => number, out?: Matrix<T>): Matrix<T> {
        return Matrix.map(this, transform, out);
    }
    /**
     * Combine this Matrix by another by applying the specified transform to pairs
     * of elements.
     *
     * The Matrices must be broadcastable. For each dimension, the sizes must
     * be equal or one of them must be 1. The resulting Matrix has the maximum
     * row and column counts of the two Matrices.
     *
     * @param that The Matrix to zip this Matrix with
     * @param transform The transform function applied to each pair of elements
     * @param out The optional Matrix to write the result to. It must have the
     * resulting broadcast shape. When provided, it is overwritten and returned
     * instead of allocating a Matrix
     */
    zip(
        that: Matrix<T>,
        transform: (x: number, y: number) => number,
        out?: Matrix<T>,
    ) {
        return Matrix.zip(this, that, transform, out);
    }

    /**
     * Reduce this Matrix to a number applying the specified accumulator
     * function on each element of the Matrix.
     *
     * @param transform
     * @param init The initial value of the number accumulated
     */
    reduce(
        transform: (acc: number, val: number) => number,
        initial: number = 0,
    ): number {
        return Matrix.reduce(this, transform, initial);
    }

    /**
     * Reduce each row of this Matrix to a column vector.
     *
     * Each row is reduced independently from left to right, starting with the
     * specified initial value.
     *
     * @param transform The accumulator function applied to each element
     * @param initial The initial value accumulated for each row
     * @param out The optional Matrix to write the result to. It must have the
     * same number of rows as this Matrix and one column. When provided, it is
     * overwritten and returned instead of allocating a Matrix
     */
    reduceRows(
        transform: (acc: number, val: number) => number,
        initial: number = 0,
        out?: Matrix<T>,
    ): Matrix<T> {
        return Matrix.reduceRows(this, transform, initial, out);
    }

    /**
     * Reduce each column of this Matrix to a row vector.
     *
     * Each column is reduced independently from top to bottom, starting with
     * the specified initial value.
     *
     * @param transform The accumulator function applied to each element
     * @param initial The initial value accumulated for each column
     * @param out The optional Matrix to write the result to. It must have one
     * row and the same number of columns as this Matrix. When provided, it is
     * overwritten and returned instead of allocating a Matrix
     */
    reduceColumns(
        transform: (acc: number, val: number) => number,
        initial: number = 0,
        out?: Matrix<T>,
    ): Matrix<T> {
        return Matrix.reduceColumns(this, transform, initial, out);
    }

    /**
     * Return a transposed view of this Matrix.
     * It neither copies nor mutates this Matrix. The returned view shares its underlying data.
     */
    transpose(copy: boolean = false): Matrix<T> {
        return Matrix.transpose(this, copy);
    }

    /**
     * Compute diagonal Matrix from row- or column vector.
     *
     * The Matrix must be either a row- or a column vector.
     */
    diagonal(): Matrix<T> {
        return Matrix.diagonal(this);
    }

    /**
     * Create a printable string representation of the Matrix.
     */
    toString(): string {
        let buffer = "";
        for (let i = 0; i < this.rowCount; i++) {
            buffer += "[";
            for (let j = 0; j < this.columnCount; j++) {
                buffer += ` ${this.get(i, j)}${
                    j == this.columnCount - 1 ? "" : ","
                }`;
            }
            buffer += " ]";
            if (i != this.rowCount - 1) {
                buffer += "\n";
            }
        }

        return buffer;
    }

    /** Create a 2D number array representation of the Matrix. */
    toRows(): number[][] {
        const out = [];
        for (let i = 0; i < this.rowCount; i++) {
            const row = [];
            for (let j = 0; j < this.columnCount; j++) {
                row.push(this.get(i, j));
            }
            out.push(row);
        }
        return out;
    }
}
