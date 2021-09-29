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
import deepnetts.net.loss.LossType;
import deepnetts.net.weights.RandomWeights;
import deepnetts.util.Tensor;
import deepnetts.util.Tensors;
import java.util.Arrays;

/**
 * Output layer of a neural network, which gives the final output of a network.
 * It is always the last layer in the network.
 *
 * @author Zoran Sevarac
 */
public class OutputLayer extends AbstractLayer {

	
	private static final long serialVersionUID = 537944573867996766L;
	
	
    protected float[] outputErrors;
    protected final String[] labels;
    protected LossType lossType;

    /**
     * Creates an instance of output layer with specified width (number of outputs)
     * and sigmoid activation function by default.
     * Outputs are labeled using generic names "Output1, 2, 3..."
     *
     * @param width layer width which represents number of network outputs
     */
    public OutputLayer(int width) {
        this.width = width;
        this.height = 1;
        this.depth = 1;

        labels = new String[depth];
        // generate default output labels
        for (int i = 0; i < depth; i++) {
            labels[i] = "out" + i;
        }

        setActivationType(ActivationType.SIGMOID);
    }

    /**
     * Creates an instance of output layer with specified width (number of outputs)
     * and specified activation function.
     * Outputs are labeled using generic names "Output1, 2, 3..."
     *
     * @param width layer width whic represents number of network outputs
     * @param actType activation function
     */
    public OutputLayer(int width, ActivationType actType) {
        this.width = width;
        this.height = 1;
        this.depth = 1;

        labels = new String[depth];
        // generate enumerated class names from 1..n
        for (int i = 0; i < depth; i++) {
            labels[i] = "Output" + i;
        }

        setActivationType(actType);
    }

    /**
     * Creates an instance of output layer with specified width (number of outputs)
     * which corresponds to number of labels and sigmoid activation function by default.
     * Outputs are labeled with strings specified in labels parameter
     *
     * @param outputLabels labels for network's outputs
     */
    public OutputLayer(String[] outputLabels) {
        this.width = outputLabels.length;
        this.height = 1;
        this.depth = 1;
        this.labels = outputLabels;
        setActivationType(ActivationType.SIGMOID);
    }

    public OutputLayer(String[] outputLabels, ActivationType actType) {
        this(outputLabels);
        setActivationType(actType);
    }

    public final void setOutputErrors(final float[] outputErrors) {
        this.outputErrors = outputErrors;
    }

    public final float[] getOutputErrors() {
        return outputErrors;
    }

    public final LossType getLossType() {
        return lossType;
    }

    public void setLossType(LossType lossType) {
        this.lossType = lossType;
    }

    @Override
    public void init() {
        inputs = prevLayer.outputs;
        outputs = new Tensor(width);
        outputErrors = new float[width];
        deltas = new Tensor(width);

        int prevLayerWidth = prevLayer.getWidth();
        weights = new Tensor(prevLayerWidth, width);
        gradients = new Tensor(prevLayerWidth, width);
        deltaWeights = new Tensor(prevLayerWidth, width);
        prevDeltaWeights = new Tensor(prevLayerWidth, width);
        RandomWeights.xavier(weights.getValues(), prevLayerWidth, width);

        biases = new float[width];
        deltaBiases = new float[width];
        prevDeltaBiases = new float[width];
        RandomWeights.randomize(biases);

    }

    /**
     * This method implements forward pass for the output layer.
     *
     * Calculates weighted input and layer outputs using sigmoid function.
     */
    @Override
    public void forward() {
        outputs.copyFrom(biases);  

        for (int outCol = 0; outCol < outputs.getCols(); outCol++) {  
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));
            }
        }

        outputs.apply(activation::getValue); 
    }

    /**
     * This method implements backward pass for the output layer.
     */
    @Override
    public void backward() {
        if (!batchMode) {   
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { 
            if (lossType == LossType.MEAN_SQUARED_ERROR) {
                final float delta = outputErrors[deltaCol] * activation.getPrime(outputs.get(deltaCol)); 
                deltas.set(deltaCol, delta);
            } else if (activationType == ActivationType.SIGMOID && lossType == LossType.CROSS_ENTROPY) { 
                deltas.set(deltaCol, outputErrors[deltaCol]); 
            } 

            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
               final float grad = deltas.get(deltaCol) * inputs.get(inCol);
               gradients.set(inCol, deltaCol, grad);

               final float deltaWeight = optim.calculateDeltaWeight(grad, inCol, deltaCol);
               deltaWeights.add(inCol, deltaCol, deltaWeight); // sum deltaWeight for batch mode
            }

            final float deltaBias = optim.calculateDeltaBias(deltas.get(deltaCol), deltaCol);
            deltaBiases[deltaCol] += deltaBias;
        }
    }

    /**
     * Applies weight changes after one learning iteration or batch
     */
    @Override
    public void applyWeightChanges() {
        if (batchMode) { 
            deltaWeights.div(batchSize);
            Tensors.div(deltaBiases, batchSize);
        }

        Tensor.copy(deltaWeights, prevDeltaWeights);
        weights.add(deltaWeights);

        Tensor.copy(deltaBiases, prevDeltaBiases);
        Tensors.add(biases, deltaBiases);

        if (batchMode) {   
            deltaWeights.fill(0);
            Tensor.fill(deltaBiases, 0);
        }
    }
    
    @Override
    public String toString() {
        return "Output Layer { width:"+width+", activation:"+activationType.name()+"}";
    }    
}