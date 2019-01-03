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
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * A collection of data set items that will be used by deep learning algorithm.
 *
 * TODO: make this class thread safe
 *
 * add a builder using .builder()   so you can build complex data set import from csv, specif columns names and stuff
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 * @param <ITEM_TYPE>
 */
public class BasicDataSet<ITEM_TYPE extends DataSetItem> implements DataSet<ITEM_TYPE> {

    /**
     * List of data set items in this data set
     */
    protected List<ITEM_TYPE> items;

    private int inputs, outputs; // number of inputs and outputs

    protected String[] columnNames; // - data set ce uvek biti importovan iz nekog fajla ili baze

    /**
     * Data set ID / name / label
     */
    private String id;

    // TODO: constructor with vector dimensions annd capacity?
    public BasicDataSet() {
        items = new ArrayList<>();
    }

    public BasicDataSet(int inputs, int outputs) {
        this();
        this.inputs = inputs;
        this.outputs = outputs;
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
        return inputs;
    }

    public int getOutputsNum() {
        return outputs;
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
     * Creates and returns data set from specified CSV file. Empty lines are
     * skipped
     *
     * @param csvFile CSV file
     * @param inputCount number of input values in a row
     * @param outputCount number of output values in a row
     * @param delimiter delimiter used to separate values
     * @return instance of data set with values loaded from file
     *
     * @throws FileNotFoundException if file was not found
     * @throws IOException if there was an error reading file
     *
     * TODO: Detect if there are labels in the first line, if there are no
     * labels, set class1, class2, class3 in classifier evaluation! and detect
     * type of attributes Move this method to some factory class or something?
     * or as a default method in data set?
     */
    public static BasicDataSet fromCSVFile(File csvFile, int inputCount, int outputCount, String delimiter) throws FileNotFoundException, IOException {
        BasicDataSet dataSet = new BasicDataSet(inputCount, outputCount);
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line = br.readLine(); // get col names from the first line
        String[] colNames = line.split(delimiter);
        while ((line = br.readLine()) != null) {
            if (line.isEmpty()) {
                continue; // skip empty lines
            }
            String[] values = line.split(delimiter);
            if (values.length != (inputCount + outputCount)) {
                throw new DeepNettsException("Wrong number of values in the row " + (dataSet.size() + 1) + ": found " + values.length + " expected " + (inputCount + outputCount));
            }
            float[] in = new float[inputCount];
            float[] out = new float[outputCount];

            try {
                // these methods could be extracted into parse float vectors
                for (int i = 0; i < inputCount; i++) { //parse inputs
                    in[i] = Float.parseFloat(values[i]);
                }

                for (int j = 0; j < outputCount; j++) { // parse outputs
                    out[j] = Float.parseFloat(values[inputCount + j]);
                }
            } catch (NumberFormatException nex) {
                throw new DeepNettsException("Error parsing line in " + (dataSet.size() + 1) + ": " + nex.getMessage(), nex);
            }

            dataSet.add(new BasicDataSetItem(in, out));
        }
        dataSet.setColumnNames(colNames);

        return dataSet;
    }
    
    public static BasicDataSet fromCSVFile(String fileName, int inputCount, int outputCount, String delimiter) throws IOException {
        return fromCSVFile(new File(fileName), inputCount, outputCount, delimiter);
    }
    
    public static BasicDataSet fromCSVFile(String fileName, int inputCount, int outputCount) throws IOException {
        return fromCSVFile(new File(fileName), inputCount, outputCount, ",");
    }
    
    // TODO: da moze da bude fromCSV ali da to bude i URL   BasicCSV.fromCSV(URL, 4, 3)
 

    /**
     * Split data set into specified number of part of equal sizes. Utility
     * method used during crossvalidation
     * note this can be default method
     * 
     * @param parts
     * @return
     */
    @Override
    public DataSet[] split(int parts) {
        double partSize = (Math.round((100d / parts)*100))/100;
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

        if (partsSum > 1) {
            throw new IllegalArgumentException("Sum of parts cannot be larger than 1!");
        }

        DataSet[] subSets = new BasicDataSet[parts.length];
        int itemIdx = 0;

        //this.shuffle(); // shuffle before splting, how to specify provide random seed? 
        for (int p = 0; p < parts.length; p++) {
            DataSet subSet = new BasicDataSet(this.inputs, this.outputs);
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
        String[] outputLabels = new String[outputs];
        for (int i = 0; i < outputs; i++) {
            outputLabels[i] = columnNames[inputs + i];
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
