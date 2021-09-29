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
package deepnetts.net.layers;

import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.weights.RandomWeights;
import deepnetts.util.Tensor;
import java.util.Arrays;

/**
 * Output layer with softmax activation function.
 *
 * @author Zoran Sevarac
 */
public class SoftmaxOutputLayer extends OutputLayer {

	
	private static final long serialVersionUID = -5557183169491335524L;
	
	
    public SoftmaxOutputLayer(int width) {
        super(width);
        setActivationType(ActivationType.SOFTMAX);
    }

    public SoftmaxOutputLayer(String[] labels) {
        super(labels);
        setActivationType(ActivationType.SOFTMAX);
    }

    @Override
    public void init() {
        inputs = prevLayer.outputs;
        outputs = new Tensor(width);
        outputErrors = new float[width];
        deltas = new Tensor(width);

        int prevLayerWidth = prevLayer.getWidth();
        weights = new Tensor(prevLayerWidth, width);
        deltaWeights = new Tensor(prevLayerWidth, width);
        gradients = new Tensor(prevLayerWidth, width);
        prevDeltaWeights = new Tensor(prevLayerWidth, width);
        RandomWeights.xavier(weights.getValues(), prevLayerWidth, width);

        biases = new float[width];
        deltaBiases = new float[width];
        prevDeltaBiases = new float[width];
        //RandomWeights.randomize(biases);
        //Tensor.fill(biases, 0.1f);
        RandomWeights.gaussian(biases, 0.1f, 0.05f);
//        RandomWeights.randomize(biases);
        
    }

    /**
     * This method implements forward pass for the output layer. Calculates
     * layer outputs using softmax function
     */
    @Override
    public void forward() {
        float maxWs = Float.NEGATIVE_INFINITY;

        for (int outCol = 0; outCol < outputs.getCols(); outCol++) {                    
            outputs.set(outCol, biases[outCol]);                                        
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {                    
                outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));    
            }

            if (outputs.get(outCol) > maxWs) { 
                maxWs = outputs.get(outCol);
            }
        }

        float denSum = 0;
        for (int col = 0; col < outputs.getCols(); col++) {
            outputs.set(col, (float) Math.exp(outputs.get(col) - maxWs)); 
            denSum += outputs.get(col);
        }

        outputs.div(denSum);
    }


    @Override
    public void backward() {
        if (!batchMode) {
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        deltas.copyFrom(outputErrors);

        for (int outCol = 0; outCol < outputs.getCols(); outCol++) { 
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) { 
                final float grad = deltas.get(outCol) * inputs.get(inCol); 
                gradients.set(inCol, outCol, grad); 
                final float deltaWeight = optim.calculateDeltaWeight(grad, inCol, outCol);   
                deltaWeights.add(inCol, outCol, deltaWeight); 
            }
            
            final float deltaBias = optim.calculateDeltaBias(deltas.get(outCol), outCol); 
            deltaBiases[outCol] += deltaBias;
        }
    }

}
