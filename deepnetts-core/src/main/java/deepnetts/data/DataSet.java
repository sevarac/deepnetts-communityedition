package deepnetts.data;

import deepnetts.util.RandomGenerator;
import java.util.Random;

/**
 * Generic interface for all data sets
 * 
 * @author Zoran Sevarac
 */
public interface DataSet <ITEM_TYPE extends DataSetItem> extends Iterable<ITEM_TYPE> {
    
    // TODO: remove idx, item, - as in List
    
    public void add(ITEM_TYPE item);
    
    public void addAll(DataSet<ITEM_TYPE> items);
    
    public ITEM_TYPE get(int index);
    
    public void clear();
    
    public boolean isEmpty();

    public int size(); 
    
    public DataSet[] split(double ... parts); 
    
    public DataSet[] split(int parts);
    
    public default DataSet[] split(long randomSeed, double ... parts) {
        RandomGenerator.getDefault().initSeed(randomSeed);
        return split(parts);
    }    
    
    public default DataSet[] split(long randomSeed, int parts) {
        RandomGenerator.getDefault().initSeed(randomSeed);
        return split(parts);
    }        
    
    public String[] getOutputLabels();
    
    public void setColumnNames(String[] labels);
    
    public void shuffle();  // shuffle using default RandomGenerator
    
}
