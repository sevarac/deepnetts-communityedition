/**
 * DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 * based learning and image recognition.
 * <p>
 * Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 * <p>
 * This file is part of DeepNetts.
 * <p>
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * <p>
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 * <p>
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */
package deepnetts.data;

import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;

import javax.visrec.ml.data.Column;
import javax.visrec.ml.data.DataSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Basic data set used for training neural networks in deep netts.
 * <p>
 * Note: implements DataSet from visrec api, and specify data set elements. Extends BasicDataSet from visrec.ml layer
 *
 * @param <E> Type of elements in this data set.
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class TabularDataSet<E extends MLDataItem> extends javax.visrec.ml.data.BasicDataSet<E> {

    private int numInputs, numOutputs; // number of inputs and outputs / target values

    protected String[] columnNames; // column names

    // TODO: do we need constructor with vector dimensions annd capacity?

    protected TabularDataSet() {
        items = new ArrayList<>();
    }

    /**
     * Create a new instance of BasicDataSet with specified size of input and output.
     *
     * @param numInputs  number of input features
     * @param numOutputs number of output features
     */
    public TabularDataSet(int numInputs, int numOutputs) {
        this();
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
    }

    public int getNumInputs() {
        return numInputs;
    }

    public int getNumOutputs() {
        return numOutputs;
    }

    /**
     * Split data set into specified number of part of equal sizes.
     * Utility method used during cross-validation
     * Note: this could  be default method
     *
     * @param parts
     * @return
     */
    @Override
    public DataSet[] split(int parts) {
        double partSize = (Math.round((100d / parts))) / 100d;
        double[] partsArr = new double[parts];
        for (int i = 0; i < parts; i++) {
            partsArr[i] = partSize;
        }

        return split(partsArr);
    }

    /**
     * Splits data set into several parts specified by the input parameter
     * partSizes. Values of partSizes parameter represent the sizes of data set
     * parts that will be returned. Part sizes are decimal values that represent
     * percents, cannot be negative or zero, and their sum must be 1
     *
     * @param parts sizes of the parts in percents
     * @return parts of the data set of specified size
     */
    @Override
    public DataSet[] split(double... parts) {
        if (parts.length < 1) {
            throw new IllegalArgumentException("Number of split parts must be greater than one");
        } else if (parts.length == 1) {
            double[] newParts = new double[2];
            newParts[0] = parts[0];
            newParts[1] = 1 - parts[0];
            parts = newParts;
        }

        double partsSum = 0;
        for (int i = 0; i < parts.length; i++) {
            if (parts[i] <= 0) {
                throw new IllegalArgumentException("Value of the part cannot be zero or negative!");
            }
            partsSum += parts[i];
        }

        if (partsSum > 1) { // a sta ako je int?
            throw new IllegalArgumentException("Sum of parts cannot be larger than 1!");
        }

        DataSet[] subSets = new TabularDataSet[parts.length];
        int itemIdx = 0;

        this.shuffle(); // shuffle before splting, using global random seed
        // todo: split so the subsets has similar distribution of values
        for (int p = 0; p < parts.length; p++) {
            TabularDataSet subSet = new TabularDataSet(this.numInputs, this.numOutputs);
            subSet.setColumnNames(this.columnNames);
            subSet.setColumns(this.getColumns());
            int itemsCount = (int) (size() * parts[p]);

            for (int j = 0; j < itemsCount; j++) {
                subSet.add(items.get(itemIdx));
                itemIdx++;
            }

            subSets[p] = subSet;
        }

        return subSets;
    }

    /**
     * Shuffles the data set items using the default random generator.
     * Default rng can be initialized independently
     */
    @Override
    public void shuffle() {
        Random rnd = RandomGenerator.getDefault().getRandom();
        Collections.shuffle(items, rnd);
    }

    /**
     * Shuffles data set items using java random generator initializes with
     * specified seed
     *
     * @param seed a seed number to initialize random generator
     * @see java.util.Random
     */
    public void shuffle(int seed) {
        Random rnd = new Random(seed);
        Collections.shuffle(items, rnd);
    }

    @Override
    public String[] getColumnNames() {
        return columnNames;
    }

    @Override
    public void setColumnNames(String[] columnNames) {
        this.columnNames = columnNames;
        List<Column> columns = new ArrayList<>(columnNames.length);
        for (int i = 0; i < columnNames.length; i++) {
            Column col = new Column(columnNames[i]);
            columns.add(col);
        }
        super.setColumns(columns);

        // Setting target columns
        String[] targetLabels = new String[numOutputs];
        for (int i = 0; i < numOutputs; i++) {
            targetLabels[i] = columnNames[numInputs + i];
        }
        setAsTargetColumns(targetLabels);
    }


    /**
     * Represents a basic data set item (single row) with input tensor and
     * target vector in a data set.
     *
     * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
     */
    public static class Item implements MLDataItem {

        private final Tensor input; // network input
        private final Tensor targetOutput; // for classifiers target can be index, int 

        public Item(float[] in, float[] targetOutput) {
            this.input = new Tensor(in);
            this.targetOutput = new Tensor(targetOutput);
        }

        public Item(Tensor input, Tensor targetOutput) {
            this.input = input;
            this.targetOutput = targetOutput;
        }

        @Override
        public Tensor getInput() {
            return input;
        }

        @Override
        public Tensor getTargetOutput() {
            return targetOutput;
        }

        public int size() {
            return input.getCols();
        }

        @Override
        public String toString() {
            return "BasicDataSetItem{" + "input=" + input + ", targetOutput=" + targetOutput + '}';
        }

    }


}
