package deepnetts.data.util;

import deepnetts.data.BasicDataSetItem;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.util.Tensor;
import java.io.Serializable;

/**
 * Performs standardization in order to get desired statistical properties of the data set.
 * Zero mean and one standard deviation.  
 * 
 * X = (X - MEAN) / STD
 * 
 * @author Zoran Sevarac
 */
public class Standardizer implements Normalizer, Serializable {
// only appy to inputs, not binary values
    private final Tensor mean;
    private final Tensor std;
    
    
    public Standardizer(DataSet<BasicDataSetItem> dataSet) {
        Tensor t = dataSet.get(0).getInput();
        // int dims = t.getDimensions();
        mean = new Tensor(t.getCols());
        std = new Tensor(t.getCols());
        
        for(BasicDataSetItem item : dataSet) {
            mean.add(item.getInput());
        }         
        mean.div((float)dataSet.size());
        
        
        //std = sqrt(sum((x-m)^2)/n);
        
        for(BasicDataSetItem item : dataSet) {
            Tensor diff = item.getInput().copy();
            diff.sub(mean); // ^2
            diff.multiplyElementWise(diff);
            std.add(diff);            
        }        
        std.div(dataSet.size()-1);
        std.sqrt();           
    }
        
    @Override
    public void normalize(DataSet<?> dataSet) {
        for (DataSetItem item : dataSet) {
            item.getInput().sub(mean);
            item.getInput().div(std);
        }
    }
    
}
