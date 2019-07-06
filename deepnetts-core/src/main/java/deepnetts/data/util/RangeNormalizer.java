package deepnetts.data.util;

import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import java.io.Serializable;

/**
 * Normalize data set to specified range.
 * Using formula X = (X-MIN) / (MAX-MIN)
 * Effectively scales all inputs and outputs to specified [MIN,MAX] range
 * Normalizes inputs and outputs.
 * 
 * @author Zoran Sevarac
 */
public class RangeNormalizer implements Normalizer, Serializable {
    private final float min;
    private final float max;
    
    /**
     * Creates a new instance of range normalizer initialized to given min and max values.
     * 
     * @param min
     * @param max 
     */
    public RangeNormalizer(float min, float max) {
        this.min = min;
        this.max = max;
    }
        
    /**
     * Performs normalization on the given inputs.
     * x = (x-min) / (max-min)
     * 
     * @param dataSet data set to normalize
     */
    @Override
    public void normalize(DataSet<?> dataSet) {
        final float divisor = max-min;
        for (DataSetItem item : dataSet) {
            item.getInput().sub(min);   // todo: how to efficently perform sub and div in one line and only one pass through tensor
            item.getInput().div(divisor);
            item.getTargetOutput().sub(min);
            item.getTargetOutput().div(divisor);
        }
    }    
}
