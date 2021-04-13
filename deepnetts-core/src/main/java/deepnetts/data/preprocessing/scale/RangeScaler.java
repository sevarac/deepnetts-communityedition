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

import java.io.Serializable;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.MLDataItem;
import javax.visrec.ml.data.preprocessing.Scaler;

/**
 * Scale data set to specified range.
 * Using formula X = (X-MIN) / (MAX-MIN)
 * Effectively scales all inputs and outputs to specified [MIN,MAX] range
 * Normalizes inputs and outputs.
 * 
 * @author Zoran Sevarac
 */
public class RangeScaler implements Scaler<DataSet<MLDataItem>>, Serializable {
    private final float min;
    private final float max;
    
    /**
     * Creates a new instance of range normalizer initialized to given min and max values.
     * 
     * @param min
     * @param max 
     */
    public RangeScaler(float min, float max) {
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
    public void apply(DataSet<MLDataItem> dataSet) {
        final float divisor = max-min;
        for (MLDataItem item : dataSet) {
            item.getInput().sub(min);   // todo: how to efficently perform sub and div in one line and only one pass through tensor
            item.getInput().div(divisor);
            item.getTargetOutput().sub(min);
            item.getTargetOutput().div(divisor);
        }
    }    
}
