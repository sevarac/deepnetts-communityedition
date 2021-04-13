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

package deepnetts.data.preprocessing.scale;

import deepnetts.util.Tensor;
import deepnetts.util.Tensors;
import java.io.Serializable;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;
import javax.visrec.ml.data.preprocessing.Scaler;

/**
 * Performs max normalization, rescales data to corresponding max value in each column.
 * Effectively scales all values to interval [0, 1]
 * Performs normalization on both inputs and outputs.
 * 
 * @author Zoran Sevarac
 */
public final class MaxScaler implements Scaler<DataSet<MLDataItem>>, Serializable {
    private Tensor maxInputs;
    private Tensor maxOutputs;
                
    /**
     * Creates a new instance of max normalizer initialized to max values in given data set.
     * 
     * @param dataSet 
     */
    public MaxScaler(DataSet<MLDataItem> dataSet) {
        // find max values for each component of input and output tensor/vector
        maxInputs = dataSet.get(0).getInput().copy();
        maxOutputs = dataSet.get(0).getTargetOutput().copy();
        
        // find max values for all components of input and output vectors
        for(MLDataItem item : dataSet) {
            maxInputs = Tensors.absMax(item.getInput(), maxInputs); 
            maxOutputs = Tensors.absMax(item.getTargetOutput(), maxOutputs);
        }        
    }
        
    /**
     * Performs max scaling on all columns of the given data set.
     * 
     * @param dataSet data set to normalize
     */
    @Override
    public void apply(DataSet<MLDataItem> dataSet) {
        // todo: prevent/catch division by zero
        for(MLDataItem item : dataSet) {
            item.getInput().div(maxInputs);   // kad je absMaxInput postao 1 1 1 1 posle pre iteracije!!!
            item.getTargetOutput().div(maxOutputs); 
        }    
    }

    public Tensor getMaxInputs() {
        return maxInputs;
    }

    public void setMaxInputs(Tensor maxInputs) {
        this.maxInputs = maxInputs;
    }

    public Tensor getMaxOutputs() {
        return maxOutputs;
    }

    public void setMaxOutputs(Tensor maxOutputs) {
        this.maxOutputs = maxOutputs;
    }
           
    /**
     * De-normalize given output vector in-place.
     * Multiplies given vector with vector used for normalization, and stores these values in same memory location as input vector.
     * 
     * @param outputs 
     */
    public void deNormalizeOutputs(final Tensor outputs) {
        outputs.multiplyElementWise(maxOutputs);
    }
    
    public void deNormalizeInputs(final Tensor inputs) {
        inputs.multiplyElementWise(maxInputs);
    }    

   
}