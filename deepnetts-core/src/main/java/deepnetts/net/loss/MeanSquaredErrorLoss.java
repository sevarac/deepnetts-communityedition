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

import deepnetts.net.NeuralNetwork;
import java.io.Serializable;

/**
 * Mean Squared Error Loss function. Sum squared errors over all patterns and
 * all outputs. Should be used for regression problems.
 *
 * @see LossFunction
 * @see CrossEntropyLoss
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public final class MeanSquaredErrorLoss implements LossFunction, Serializable {

    private final float[] outputError;
    private float totalError = 0;
    private int patternCount = 0;
    private float regularizationSum = 0;
    
    /**
     * Creates a new mean squared error loss for the given neural network.
     * 
     * @param neuralNet 
     */
    public MeanSquaredErrorLoss(NeuralNetwork neuralNet) {
        outputError = new float[neuralNet.getOutputLayer().getWidth()];
    }

    /**
     * Adds output error vector for the given predicted and target output vectors
     * to total error sum and returns and  error vector.
     *
     * @param predictedOutput   
     * @param targetOutput  
     * @return
     */
    @Override
    public float[] addPatternError(final float[] predictedOutput, final float[] targetOutput) {
        for (int i = 0; i < predictedOutput.length; i++) {
            outputError[i] = predictedOutput[i] - targetOutput[i];
            totalError += outputError[i] * outputError[i];
        }

        patternCount++;
        return outputError;
    }

    @Override
    public void addRegularizationSum(final float regSum) {
       // regularizationSum = regSum;
    }    
    
    @Override
    public float getTotal() {
        return ( totalError / (2 * patternCount * outputError.length ) ) + regularizationSum; // should we use patternCount * outputCount in denominator ??? confirm this
    }

    @Override
    public void reset() {
        totalError = 0;
        patternCount = 0;
        regularizationSum = 0;
    }

}
