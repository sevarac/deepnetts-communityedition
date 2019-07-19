package deepnetts.data.util;

import deepnetts.data.BasicDataSetItem;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.util.Tensors;
import java.io.Serializable;

/**
 * Decimal scale normalization for the given data set.
 * Effectively dives all inputs and outputs with 10^k so they are all scaled to [0,1]
 * 
 * @author Zoran Sevarac
 */
public class DecimalScaleNormalizer implements Normalizer, Serializable {
    private float[] inputDivisor;   // TODO: make these tensors
    private float[] outputDivisor;
                
    /**
     * Creates a new instance of max normalizer initialized to max values in given data set.
     * 
     * @param dataSet 
     */
    public DecimalScaleNormalizer(DataSet<BasicDataSetItem> dataSet) {
        // find max values for each component of input and output tensor/vector
        inputDivisor = dataSet.get(0).getInput().copy().getValues();
        outputDivisor = dataSet.get(0).getTargetOutput().copy().getValues();
        
        // nadji maksimume za sve kolone
        for(BasicDataSetItem item : dataSet) {
            inputDivisor = Tensors.absMax(item.getInput().getValues(), inputDivisor);
            outputDivisor = Tensors.absMax(item.getTargetOutput().getValues(), outputDivisor);
        }   
        
        // onda za svaki vektor nadji decimalnu skalu, 1, 10, 100, 1000 while (x>1) { x = x / 10.0f; scale++;}        
        for(BasicDataSetItem item : dataSet) { 
            // i za svaku komponentu ulaznog ili izlaznog vektora
            inputDivisor = getDecimalScaleFor(item.getInput().getValues());
            outputDivisor = getDecimalScaleFor(item.getTargetOutput().getValues());        
        }
                
    }
    
      private float[] getDecimalScaleFor(float[] values) {
          float[] decimalScales = new float[values.length];
          
          for(int i=0; i<values.length; i++) {
              decimalScales[i] = getDecimalScaleFor(values[i]);
          }
          
          return decimalScales;
      }
    
    private float getDecimalScaleFor(float val) {
        float scale = 1;
        val = Math.abs(val);
        
         while (val>1) {
             scale = scale * 10;
             val = val / scale; 
         }         
        return scale; 
    }
        
    /**
     * Performs normalization on the given inputs.
     * 
     * @param dataSet data set to normalize
     */
    @Override
    public void normalize(DataSet<?> dataSet) {
        for(DataSetItem item : dataSet) {
            item.getInput().div(inputDivisor);
            item.getTargetOutput().div(outputDivisor);
        }    
    }
       
}