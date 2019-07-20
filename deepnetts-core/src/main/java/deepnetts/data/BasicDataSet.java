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
package deepnetts.data;

import deepnetts.util.DeepNettsException;
import deepnetts.util.RandomGenerator;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * A collection of data set items that will be used by deep learning algorithm.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 * @param <ITEM_TYPE>
 */
public class BasicDataSet<ITEM_TYPE extends DataSetItem> implements DataSet<ITEM_TYPE> {

    /**
     * List of data set items in this data set
     */
    protected List<ITEM_TYPE> items;

    private int inputsNum, outputsNum; // number of inputs and outputs

    protected String[] columnNames; // - data set ce uvek biti importovan iz nekog fajla ili baze

    /**
     * Data set ID / name / label
     */
    private String id;

    // TODO: constructor with vector dimensions annd capacity?
    protected BasicDataSet() {
        items = new ArrayList<>();
    }

    /**
     * Create a new instance of BasicDataSet with specified length of input and output.
     * @param inputsNum number of input features
     * @param outputsNum number of output features
     */
    public BasicDataSet(int inputsNum, int outputsNum) {
        this();
        this.inputsNum = inputsNum;
        this.outputsNum = outputsNum;
    }

    @Override
    public Iterator<ITEM_TYPE> iterator() {
        return items.iterator();
    }

    @Override
    public void add(ITEM_TYPE item) {
        items.add(item);
    }

    @Override
    public ITEM_TYPE get(int index) {
        return items.get(index);
    }

    @Override
    public void clear() {
        items.clear();
    }

    @Override
    public boolean isEmpty() {
        return items.isEmpty();
    }

    @Override
    public int size() {
        return items.size();
    }

    public int getInputsNum() {
        return inputsNum;
    }

    public int getOutputsNum() {
        return outputsNum;
    }

    public List<ITEM_TYPE> getItems() {
        return Collections.unmodifiableList(items);
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
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
        double partSize = (Math.round((100d / parts)))/100d;
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
            throw new IllegalArgumentException("");
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

        DataSet[] subSets = new BasicDataSet[parts.length];
        int itemIdx = 0;

        this.shuffle(); // shuffle before splting, using global random seed
        for (int p = 0; p < parts.length; p++) {
            DataSet subSet = new BasicDataSet(this.inputsNum, this.outputsNum);
            subSet.setColumnNames(this.columnNames);
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

    public String[] getColumnNames() {
        return columnNames;
    }

    @Override
    public void setColumnNames(String[] columnNames) {
        this.columnNames = columnNames;
    }

    @Override
    public String[] getOutputLabels() {
        String[] outputLabels = new String[outputsNum];
        for (int i = 0; i < outputsNum; i++) {
            outputLabels[i] = columnNames[inputsNum + i];
        }

        return outputLabels;
    }

    @Override
    public void addAll(DataSet<ITEM_TYPE> moreItems) {
        for (ITEM_TYPE item : moreItems) {
            items.add(item);
        }
    }
}