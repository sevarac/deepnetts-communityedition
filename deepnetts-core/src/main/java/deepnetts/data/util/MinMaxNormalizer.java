package deepnetts.data.util;

import deepnetts.data.BasicDataSetItem;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.util.Tensor;
import deepnetts.util.Tensors;
import java.io.Serializable;

/**
 * Performs Min Max normalization on the given data set.
 * 
 * @author Zoran Sevarac
 */
public class MinMaxNormalizer implements Normalizer, Serializable {
    private Tensor minInput;
    private Tensor maxInput;
    private Tensor minOutput;   // ovo ne bi trebalo na outputs da se primenjuje???
    private Tensor maxOutput;   
    
    /**
     * Creates a new instance of max normalizer initialized to max values in given data set.
     * 
     * @param dataSet 
     */
    public MinMaxNormalizer(DataSet<BasicDataSetItem> dataSet) {
        // find min and max values for each component of input and output tensor/vector
        minInput = dataSet.get(0).getInput().copy();
        maxInput = dataSet.get(0).getInput().copy();
        minOutput = dataSet.get(0).getTargetOutput().copy();
        maxOutput = dataSet.get(0).getTargetOutput().copy();
        
        for(BasicDataSetItem item : dataSet) {
            minInput = Tensors.absMin(item.getInput(), minInput);
            maxInput = Tensors.absMax(item.getInput(), maxInput);
            minOutput = Tensors.absMin(item.getTargetOutput(), minOutput);
            maxOutput = Tensors.absMax(item.getTargetOutput(), maxOutput);
        }        
    }
        
    /**
     * Performs normalization on the given inputs.
     * x = (x-min) / (max-min)
     * 
     * @param dataSet data set to normalize
     */
    @Override
    public void normalize(DataSet<?> dataSet) {
        Tensor inDivider = maxInput.copy();
        maxInput.sub(minInput);

        Tensor outDivider = maxOutput.copy();
        maxOutput.sub(minOutput);

        for (DataSetItem item : dataSet) {
            item.getInput().sub(minInput);
            item.getInput().div(inDivider);
            item.getTargetOutput().sub(minInput);
            item.getTargetOutput().sub(outDivider);
        }
    }
}
