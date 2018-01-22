/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
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
 * DataSet should also be an interface like List or Collection.
 * We can have BasicDataSet or DefaultDataSet
 * 
 * this should be the interface in visrec ml
 * TODO: make this class thread safe
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 * @param <ITEM_TYPE>
 */
public class DataSet<ITEM_TYPE extends DataSetItem> implements Iterable<ITEM_TYPE> {
    protected List<ITEM_TYPE> items;
    private Iterator<ITEM_TYPE> iterator;
    
    private int inputs, outputs;

    private String label; // label should be removed probably
     
    // TODO: constructor with vector dimensions annd capacity?
     
    public DataSet() {
        items = new ArrayList<>();
    }

    public DataSet(int inputs, int outputs) {
        this();
        this.inputs = inputs;
        this.outputs = outputs;
    }     
               
    @Override
    public Iterator<ITEM_TYPE> iterator() {
        iterator =  items.iterator(); // is this thread safe?
        return iterator;
    }

    public void add(ITEM_TYPE item) {
        items.add(item);
    }
    
    public ITEM_TYPE get(int index) {
        return items.get(index);
    }

    public int size() {
        return items.size();
    }
    
    public boolean isEmpty() {
        return items.isEmpty();
    }
    
    public List<ITEM_TYPE> getItems() {
        return Collections.unmodifiableList(items);
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }
    
    
    /**
     * Creates and returns data set from specified CSV file.
     * Empty lines are skipped
     * 
     * @param csvFile CSV file
     * @param inputCount number of input values in a row
     * @param outputCount number of output values in a row
     * @param delimiter delimiter used to separate values
     * @return instance of data set with values loaded from file
     * 
     * @throws FileNotFoundException if file was not found
     * @throws IOException  if there was an error reading file
     * 
     * error messages - specific line line num , row
     */
    public static DataSet fromCSVFile(File csvFile, int inputCount, int outputCount, String delimiter) throws FileNotFoundException, IOException {
        DataSet dataSet = new DataSet(inputCount, outputCount);
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line = "";
        while((line = br.readLine()) != null) {
           if (line.isEmpty()) continue; // skip empty lines
           String[] values = line.split(delimiter);
           if (values.length != (inputCount + outputCount)) throw new DeepNettsException("Wrong number of values in the row "+(dataSet.size()+1) + ": found "+values.length+ " expected "+(inputCount + outputCount));
           float[] in =  new float[inputCount];
           float[] out = new float[outputCount];
           
           try {
            // these methods could be extracted into parse float vectors
            for(int i=0; i<inputCount; i++) { //parse inputs
                in[i] = Float.parseFloat(values[i]);      
            }

            for(int j=0; j<outputCount; j++) { // parse outputs
                out[j] = Float.parseFloat(values[inputCount+j]);
            }
           } catch(NumberFormatException nex) {
              throw new DeepNettsException("Error parsing line in "+(dataSet.size()+1)+ ": "+nex.getMessage(), nex); 
           }
           
           dataSet.add(new BasicDataSetItem(in, out));                      
        }
                
        return dataSet;          
    }
        
    /**
     * Splits data set into several parts specified by the input parameter partSizes.
     * Values of partSizes parameter represent the sizes of data set parts that will be returned.
     * Part sizes are integer values that represent percents, cannot be negative or zero, and their sum must be 100
     * 
     * @param partSizes sizes of the parts in percents
     * @return parts of the data set of specified size 
     */
    public DataSet[] split(int ... partSizes) {    
        if (partSizes.length < 2) throw new IllegalArgumentException("Must specify at least two parts");
        int partsSum=0;
        for(int i=0; i<partSizes.length; i++) {
            if (partSizes[i] <= 0) throw new IllegalArgumentException("Value of the part cannot be zero or negative!");
            partsSum += partSizes[i];
        }
        
        if (partsSum > 100) throw new IllegalArgumentException("Sum of parts cannot be larger than 100!");
                
        DataSet[] subSets = new DataSet[partSizes.length];
        int itemIdx=0;
        
        for(int p = 0; p < partSizes.length; p++) {
             DataSet subSet = new DataSet(this.inputs, this.outputs); 
             int itemsCount =(int) (size() * partSizes[p] / 100.0f);
             
             for(int j=0; j<itemsCount; j++) {
                 subSet.add(items.get(itemIdx));
                 itemIdx++;
             }
             
             subSets[p] = subSet;
        }
                        
        return subSets;
    }

    /**
     * Shuffles the data set items using the default random generator
     */
    public void shuffle() {
        Random rnd = RandomGenerator.getDefault().getRandom();
        Collections.shuffle(items, rnd); // use one with rand param
    }
    
    /**
     * Shuffles data set items using java random generator initializes with specified seed
     * 
     * @param seed a seed number to initialize random generator
     * @see java.util.Random
     */
    public void shuffle(int seed) {
        Random rnd = new Random(seed);
        Collections.shuffle(items, rnd);
    }        
    
            
}
