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
package deepnetts.net.loss;

import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.NeuralNetwork;

/**
 * Interface for all loss functions.
 *
 * TODO: add method to return firts derivative by y : dE/dy ?
 *
 * @see MeanSquaredErrorLoss
 * @see CrossEntropyLoss
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public interface LossFunction {

    /**
     * Calculates pattern error for the specified predicted and target outputs,
     * adds the error to total error, and returns the pattern error.
     *
     * @param predictedOutput actual network output
     * @param targetOutput target network output
     * @return output error vector
     */
    public float[] addPatternError(float[] predictedOutput, float[] targetOutput);
    
    /**
     * Adds regularization sum to loss function
     */
    public void addRegularizationSum(float regSum);
       
    /**
     * Returns the total error calculated by this loss function.
     *
     * @return total error calculated by this loss function
     */
    public float getTotal();

    /**
     * Resets the total error and pattern counter.
     */
    public void reset();
    
    /**
     * Calculates and returns loss function value for the given neural network and test set.
     * 
     * @param nnet
     * @param testSet
     * @return 
     */
    default public float valueFor(NeuralNetwork nnet, DataSet<? extends DataSetItem> testSet) {
        for(DataSetItem tsItem : testSet) {            
            nnet.setInput(tsItem.getInput());
            float[] output = nnet.getOutput();
            addPatternError(output, tsItem.getTargetOutput());
        }
        return getTotal();
    }

}
