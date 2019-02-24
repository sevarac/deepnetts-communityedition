/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */
package deepnetts.util;

import java.io.Serializable;
import java.util.Arrays;
import java.util.function.Function;

/**
 * This class represents multidimensional array/matrix (can be 1D, 2D, 3D or
 * 4D). https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
 *
 * @author Zoran Sevarac
 */
public class Tensor implements Serializable {

    // tensor dimensions - better use shape
    private final int cols, rows, depth, fourthDim, dimensions;
    // todo & benchmark shape max dimenzije 4 da podrzi na postojece atribute
    private final int[] shape = new int[4];
    private int rank; // how many dimensions in shape - shape.length ; same as dimensions
    private int size; // broj elemenata u nizu - kad se izmnoze sve dimenzije
    // dimensions = shape.length;
    // column first order
    // poslednja dimenzija se uvek najbrze vrti, prva najsporije (kao i kod prirodnih brojeva)
    // ako je poslednja dimenzija najsporija onda dodatne dimenzije idu napred
    // col - shape[0]
    // row, col - shape[1], shape[0] ? ili shape[0], shape[1] ? posto je poslednji najbrzi column je poslednjii treba da bude najbrzi
    // depth, row, col
    // forth, depth, row, col   -    ato je sve za column first, odnosno kolona je najbrzadimenzija i tako bih resio ono sa DenseLayer i weights indexom

    // u ndim indexu col je 0, row je 1, z je 2, fourth je 3 (to je stride)
    // good explanation and formulas for indexing
    // https://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
    /**
     * Values stored in this tensor make it final , only input layer and tests
     * sets values
     */
    private float values[]; // todo: use ByteBuffer instead of array in order to avoid range checking

    /**
     * Creates a single row tensor with specified values.
     *
     * @param values values of column tensor
     */
    public Tensor(final float... values) {
        this.rows = 1;
        this.cols = values.length;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 1;
        this.values = values;
    }

    /**
     * Creates a 2D tensor / matrix with specified values.
     *
     * @param vals
     */
    public Tensor(final float[][] vals) {
        this.rows = vals.length;
        this.cols = vals[0].length;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 2;
        this.values = new float[rows * cols];

        // copyFrom array to single dim
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                set(row, col, vals[row][col]);
            }
        }
    }

    /**
     * Creates a 3D tensor from specified 3D array
     *
     * @param vals 2D array of tensor values
     */
    public Tensor(final float[][][] vals) {
        this.depth = vals.length;
        this.rows = vals[0].length;
        this.cols = vals[0][0].length;

        this.fourthDim = 1;
        this.dimensions = 3;
        this.values = new float[rows * cols * depth];

        // copyFrom array
        for (int z = 0; z < depth; z++) {
            for (int row = 0; row < rows; row++) {
                for (int col = 0; col < cols; col++) {
                    set(row, col, z, vals[z][row][col]);
                }
            }
        }
    }

    public Tensor(final float[][][][] vals) {

        this.fourthDim = vals.length;
        this.depth = vals[0].length;
        this.rows = vals[0][0].length;
        this.cols = vals[0][0][0].length;

        this.dimensions = 4;
        this.values = new float[rows * cols * depth * fourthDim];

        // copyFrom array
        for (int f = 0; f < fourthDim; f++) {
            for (int z = 0; z < depth; z++) {
                for (int row = 0; row < rows; row++) {
                    for (int col = 0; col < cols; col++) {
                        set(row, col, z, f, vals[f][z][row][col]);
                    }
                }
            }
        }
    }

    /**
     * Creates an empty single row tensor with specified number of columns.
     *
     * @param cols number of columns
     */
    public Tensor(int cols) {
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }

        this.cols = cols;
        this.rows = 1;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 1;
        values = new float[cols];
    }

    // ovaj najbolje preko factory metode
    public Tensor(int cols, float val) {
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }

        this.cols = cols;
        this.rows = 1;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 1;
        values = new float[cols];

        for (int i = 0; i < values.length; i++) {
            values[i] = val;
        }
    }

    /**
     * Creates a tensor with specified number of rows and columns.
     *
     * @param rows number of rows
     * @param cols number of columns
     */
    public Tensor(int rows, int cols) {
        if (rows < 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }

        this.rows = rows;
        this.cols = cols;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 2;
        values = new float[rows * cols];
    }

    public Tensor(int rows, int cols, float[] values) {
        if (rows < 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }
        if (rows * cols != values.length) {
            throw new IllegalArgumentException("Number of values does not match tensor dimensions! " + values.length);
        }

        this.rows = rows;
        this.cols = cols;
        this.depth = 1;
        this.fourthDim = 1;
        this.dimensions = 2;

        this.values = values;
    }

    /**
     * Creates a 3D tensor with specified number of rows, cols and depth.
     *
     * @param rows number of rows
     * @param cols number of columns
     * @param depth tensor depth
     */
    public Tensor(int rows, int cols, int depth) { // trebalo bi depth, rows, cols
        if (rows < 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }
        if (depth < 0) {
            throw new IllegalArgumentException("Depth cannot be negative: " + depth);
        }

        this.rows = rows;
        this.cols = cols;
        this.depth = depth;
        this.fourthDim = 1;
        this.dimensions = 3;
        this.values = new float[rows * cols * depth];
    }

    // cols, rows, 3rd, 4th?
    public Tensor(int rows, int cols, int depth, int fourthDim) { // trebalo bi fourthDim, depth, rows, cols
        if (rows < 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }
        if (depth < 0) {
            throw new IllegalArgumentException("Depth cannot be negative: " + depth);
        }
        if (fourthDim < 0) {
            throw new IllegalArgumentException("fourthDim cannot be negative: " + fourthDim);
        }

        this.rows = rows;
        this.cols = cols;
        this.depth = depth;
        this.fourthDim = fourthDim;
        this.dimensions = 4;
        this.values = new float[rows * cols * depth * fourthDim];
    }

    public Tensor(int rows, int cols, int depth, int fourthDim, float[] values) {
        if (rows < 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }
        if (depth < 0) {
            throw new IllegalArgumentException("Depth cannot be negative: " + depth);
        }
        if (fourthDim < 0) {
            throw new IllegalArgumentException("fourthDim cannot be negative: " + fourthDim);
        }

        this.rows = rows;
        this.cols = cols;
        this.depth = depth;
        this.fourthDim = fourthDim;
        this.dimensions = 4;
        this.values = values;
    }

    public Tensor(int rows, int cols, int depth, float[] values) {
        if (rows < 0) {
            throw new IllegalArgumentException("Number of rows cannot be negative: " + rows);
        }
        if (cols < 0) {
            throw new IllegalArgumentException("Number of cols cannot be negative: " + cols);
        }
        if (depth < 0) {
            throw new IllegalArgumentException("Depth cannot be negative: " + depth);
        }
        if (rows * cols * depth != values.length) {
            throw new IllegalArgumentException("Number of values does not match tensor dimensions! " + values.length);
        }

        this.cols = cols;
        this.rows = rows;
        this.depth = depth;
        this.fourthDim = 1;
        this.dimensions = 3;
        this.values = values;
    }

//    public Tensor(float[] values, int ... shape ) {
//        if (shape.length > 4) throw new IllegalArgumentException("Tensor can have max 4 dimensions");
//        this.dimensions = shape.length;
//
//        if (dimensions == 1) this.cols = shape[1];
//
//        this.rows = shape[0];
//
//        if (dimensions > 2)
//
//        this.depth = shape[2];
//        this.fourthDim = shape[3];
//
//        this.values = values;
//    }
    private Tensor(Tensor t) {
        this.cols = t.cols;
        this.rows = t.rows;
        this.depth = t.depth;
        this.fourthDim = t.fourthDim;
        this.dimensions = t.dimensions;
        values = new float[t.values.length];

        System.arraycopy(t.values, 0, values, 0, t.values.length);
    }

    /**
     * Gets value at specified index position.
     *
     * @param idx
     * @return
     */
    public final float get(final int idx) {
        return values[idx];
    }

    /**
     * Sets value at specified index position.
     *
     * @param idx
     * @param val
     * @return
     */
    public final float set(final int idx, final float val) {
        return values[idx] = val;
    }

    // make sure this method gets inlined - final?   keeping hot methods small (35 bytecodes or less) final migh help, it will get inlined if its a hotspot - frequent calls
    /**
     * Returns matrix value at row, col
     *
     * @param col
     * @param row
     * @return value at [row, col]
     */
    public final float get(final int row, final int col) {
        final int idx = row * cols + col;
        return values[idx];
    }

    /**
     * Sets matrix value at specified [row, col] position
     *
     * @param row matrix roe
     * @param col matrix col
     * @param val value to set
     */
    public final void set(final int row, final int col, final float val) {
        final int idx = row * cols + col;
        values[idx] = val;
    }

    /**
     * Returns value at row, col, z
     *
     * @param col
     * @param row
     * @param z
     * @return
     */
    public final float get(final int row, final int col, final int z) {
        final int idx = z * cols * rows + row * cols + col;
        return values[idx];
    }

    public final void set(final int row, final int col, final int z, final float val) {
        final int idx = z * cols * rows + row * cols + col;
        values[idx] = val;
    }

    public final float get(final int row, final int col, final int z, final int fourth) {
        final int idx = fourth * rows * cols * depth + z * rows * cols + row * cols + col;
        return values[idx];
    }

    public final void set(final int row, final int col, final int z, final int fourth, final float val) {
        final int idx = fourth * rows * cols * depth + z * rows * cols + row * cols + col;
        values[idx] = val;
    }

    // still under development dont use it!
    // pretpostavi d je dim
    public final float getWithStride(final int[] idxs) {
        // final int idx = idxs[3] * shape[2] * shape[1] * shape[0] + idxs[2] * shape[1] * shape[0] + idxs[1] * shape[0] + idxs[0];
        // final int idx = fourth * rows * cols * depth             + z * rows * cols               + row * cols         + col;
        final int idx = idxs[0] * shape[1] * shape[2] * shape[3] + idxs[1] * shape[2] * shape[3] + idxs[2] * shape[3] + idxs[3];
        return values[idx];
    }

    public final float[] getValues() {
        return values;
    }

    public final void setValues(final float... values) {
//        if (values.length != this.values.length) throw new DeepNettsException("Arrays are not of same size!");
        this.values = values;
    }

    public final void copyFrom(final float[] src) {
        System.arraycopy(src, 0, values, 0, values.length);
    }

    public final int getCols() {
        return cols;
    }

    public final int getRows() {
        return rows;
    }

    public final int getDepth() {
        return depth;
    }

    public final int getFourthDim() {
        return fourthDim;
    }

    public final int getDimensions() {
        return dimensions;
    }

    public final int size() {
        return values.length;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("[");
        for (int i = 0; i < values.length; i++) {
            sb.append(values[i]);
            if ( ((i+1) % cols == 0) && (i < values.length - 1)  ) {
                sb.append("; ");
            } else if (i < values.length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");

        return sb.toString();
    }

    public final void add(final int idx, final float value) {
        values[idx] += value;
    }

    /**
     * Adds specified value to matrix value at position x, y
     *
     * @param col
     * @param row
     * @param value
     */
    public final void add(final int row, final int col, final float value) {
        final int idx = row * cols + col;
        values[idx] += value;
    }

    public final void add(final int row, final int col, final int z, final float value) {
        final int idx = z * cols * rows + row * cols + col;
        values[idx] += value;
    }

    public final void add(final int row, final int col, final int z, final int fourth, final float value) {
        final int idx = fourth * cols * rows * depth + z * cols * rows + row * cols + col;
        values[idx] += value;
    }



    /**
     * Adds specified tensor t to this tensor.
     *
     * @param t tensor to add
     */
    public final void add(Tensor t) {
        for (int i = 0; i < values.length; i++) {
            values[i] += t.values[i];
        }
    }

    public final void sub(final int row, final int col, final float value) {
        final int idx = row * cols + col;
        values[idx] -= value;
    }

    public final void sub(final int row, final int col, final int z, final float value) {
        final int idx = z * rows * cols + row * cols + col;
        values[idx] -= value;
    }

    public final void sub(final int row, final int col, final int z, final int fourth, final float value) {
        final int idx = fourth * rows * cols * depth + z * rows * cols + row * cols + col;
        values[idx] -= value;
    }

    /**
     * Subtracts specified tensor t from this tensor.
     *
     * @param t tensor to subtract
     */
    public final void sub(final Tensor t) {
        for (int i = 0; i < values.length; i++) {
            values[i] -= t.values[i];
        }
    }

    /**
     * Subtracts tensor t2 from t1. The result is t1.
     *
     * @param t1
     * @param t2
     */
    public final static void sub(final Tensor t1, final Tensor t2) {
        for (int i = 0; i < t1.values.length; i++) {
            t1.values[i] -= t2.values[i];
        }
    }

    /**
     * Divide all values in this tensor with specified value.
     *
     * @param value
     */
    public final void div(final float value) {
        for (int i = 0; i < values.length; i++) {
            values[i] /= value;
        }
    }

    /**
     * Fills the entire tensor with specified value.
     *
     * @param value value used to fill tensor
     */
    public final void fill(final float value) {
        for (int i = 0; i < values.length; i++) {
            values[i] = value;
        }
    }

    public static final void fill(final float[] array, final float val) {
        for (int i = 0; i < array.length; i++) {
            array[i] = val;
        }
    }

    public static void div(final float[] array, final float val) {
        for (int i = 0; i < array.length; i++) {
            array[i] /= val;
        }
    }

    public static final void sub(final float[] array1, final float[] array2) {
        for (int i = 0; i < array1.length; i++) {
            array1[i] -= array2[i];
        }
    }

    public static final void add(final float[] array1, final float[] array2) {
        for (int i = 0; i < array1.length; i++) {
            array1[i] += array2[i];
        }
    }

    public static final void copy(final Tensor src, final Tensor dest) {
        System.arraycopy(src.values, 0, dest.values, 0, src.values.length);
    }

    public static final void copy(final float[] src, final float[] dest) {
        System.arraycopy(src, 0, dest, 0, src.length);
    }

    public void apply(Function<Float, Float> f) {
        for(int i=0; i<values.length; i++) {
            values[i] = f.apply(values[i]);
        }
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Tensor other = (Tensor) obj;
        if (this.cols != other.cols) {
            return false;
        }
        if (this.rows != other.rows) {
            return false;
        }
        if (this.depth != other.depth) {
            return false;
        }
        if (this.fourthDim != other.fourthDim) {
            return false;
        }
        if (this.dimensions != other.dimensions) {
            return false;
        }
        if (!Arrays.equals(this.values, other.values)) {
            return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int hash = 3;
        hash = 41 * hash + this.cols;
        hash = 41 * hash + this.rows;
        hash = 41 * hash + this.depth;
        hash = 41 * hash + this.fourthDim;
        hash = 41 * hash + this.dimensions;
        hash = 41 * hash + Arrays.hashCode(this.values);
        return hash;
    }

    public boolean equals(Tensor t2, float delta) {
        float[] arr2 = t2.getValues();

        for (int i = 0; i < values.length; i++) {
            if (Math.abs(values[i] - arr2[i]) > delta) {
                return false;
            }
        }
        return true;
    }

// add clone using apache clone builder

    public static String valuesAsString(Tensor[] tensors) {
        StringBuilder sb = new StringBuilder();

        for (Tensor t : tensors) {
            sb.append(t.toString());
        }

        return sb.toString();
    }

    /**
     * Sets tensor values from csv string.
     *
     * @param values csv string with values
     */
    public void setValuesFromString(String values) {
        String[] strArr = values.split(",");
        for (int i = 0; i < strArr.length; i++) {
            this.values[i] = Float.parseFloat(strArr[i]);
        }
    }

    /**
     * Factory method for creating tensor instance,
     *
     * @param rows
     * @param cols
     * @param values
     * @return
     */
    public static Tensor create(int rows, int cols, float[] values) {
        return new Tensor(rows, cols, values);
    }

    public static Tensor create(int rows, int cols, int depth, float[] values) {
        return new Tensor(rows, cols, depth, values);
    }

    public static Tensor create(int rows, int cols, int depth, int fourthDim, float[] values) {
        return new Tensor(rows, cols, depth, fourthDim, values);
    }

    /**
     * Returns sum of abs values of this tensor - L1 norm
     *
     * @return L1 norm
     */
    public float sumAbs() {
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += Math.abs(values[i]);
        }
        return sum;
    }

    /**
     * Returns sum of sqr values of this tensor - L2 norm
     *
     * @return L2 norm
     */
    public float sumSqr() {
        float sum = 0;
        for (int i = 0; i < values.length; i++) {
            sum += values[i] * values[i];
        }
        return sum;
    }

    // works for 2d tensors
    public void randomize() {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                values[r * cols + c] = RandomGenerator.getDefault().nextFloat();
            }
        }
    }

    public void multiplyElementWise(Tensor tensor2) {
        for(int i=0; i<values.length; i++) {
            values[i] *= tensor2.values[i];
        }
    }



}
