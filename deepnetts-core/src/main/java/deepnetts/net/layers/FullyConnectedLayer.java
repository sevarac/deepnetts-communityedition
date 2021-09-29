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
import deepnetts.core.DeepNetts;
import deepnetts.net.layers.activation.ActivationFunction;
import deepnetts.net.weights.RandomWeights;
import deepnetts.util.Tensor;
import java.util.Arrays;
import java.util.logging.Logger;
import deepnetts.util.DeepNettsException;
import deepnetts.util.Tensors;

/**
 * Fully connected layer is used as hidden layer in the neural network, and it
 * has a single row of units/nodes/neurons connected to all neurons in previous
 * and next layer.
 *
 * @see ActivationType
 * @see ActivationFunction
 * @author Zoran Sevarac
 */
public final class FullyConnectedLayer extends AbstractLayer {

	
	private static final long serialVersionUID = -8383673021557469094L;
	
	
    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    /**
     * Creates an instance of fully connected layer with specified width (number
     * of neurons) and sigmoid activation function.
     *
     * @param width layer width / number of neurons in this layer
     */
    public FullyConnectedLayer(int width) {
        this.width = width;
        this.height = 1;
        this.depth = 1;

        setActivationType(ActivationType.SIGMOID);
    }

    /**
     * Creates an instance of fully connected layer with specified width (number
     * of neurons) and activation function type.
     *
     * @param width layer width / number of neurons in this layer
     * @param actType activation function type to use in this layer
     * @see ActivationFunctions
     */
    public FullyConnectedLayer(int width, ActivationType actType) {
        this(width);
        setActivationType(actType);
    }

    /**
     * Creates all internal data structures: inputs, weights, biases, outputs,
     * deltas, deltaWeights, deltaBiases prevDeltaWeights, prevDeltaBiases. Init
     * weights and biases. This method is called from network builder during
     * initialization
     */
    @Override
    public void init() {
        if (!(prevLayer instanceof InputLayer
                || prevLayer instanceof FullyConnectedLayer
                || prevLayer instanceof MaxPoolingLayer
                || prevLayer instanceof ConvolutionalLayer)) {
            throw new DeepNettsException("Bad network architecture! Fully Connected Layer can be connected only to Input, FullyConnected, Maxpooling or Convolutional layer as previous layer.");
        }

        if (!(nextLayer instanceof FullyConnectedLayer
                || nextLayer instanceof OutputLayer)) {
            throw new DeepNettsException("Bad network architecture! Fully Connected Layer can only be connected only to Fully Connected Layer or Output layer as next layer");
        }

        inputs = prevLayer.outputs;
        outputs = new Tensor(width);
        deltas = new Tensor(width);

        if (prevLayer instanceof FullyConnectedLayer || (prevLayer instanceof InputLayer && prevLayer.height == 1 && prevLayer.depth == 1)) { // ovo ako je prethodni 1d layer, odnosno ako je prethodni fully connected
            weights = new Tensor(prevLayer.width, width);
            deltaWeights = new Tensor(prevLayer.width, width);
            gradients = new Tensor(prevLayer.width, width);
            prevDeltaWeights = new Tensor(prevLayer.width, width);

            prevGradSqrSum = new Tensor(prevLayer.width, width);
            prevDeltaWeightSqrSum = new Tensor(prevLayer.width, width);
            prevBiasSqrSum = new Tensor(width);
            prevDeltaBiasSqrSum = new Tensor(width);

            if (activationType == ActivationType.RELU || activationType == ActivationType.LEAKY_RELU) {
                RandomWeights.he(weights.getValues(), outputs.size());
            } else {    // sigmoid tanh
                RandomWeights.xavier(weights.getValues(), prevLayer.width, width);
            }

        } else if ((prevLayer instanceof MaxPoolingLayer) || (prevLayer instanceof ConvolutionalLayer) || (prevLayer instanceof InputLayer)) {
            weights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            deltaWeights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            gradients = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            prevDeltaWeights = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);

            prevGradSqrSum = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width);
            prevBiasSqrSum = new Tensor(width);

            prevDeltaWeightSqrSum = new Tensor(prevLayer.width, prevLayer.height, prevLayer.depth, width); // ada delta
            prevDeltaBiasSqrSum = new Tensor(width);

            int totalInputs = prevLayer.getWidth() * prevLayer.getHeight() * prevLayer.getDepth();

            if (activationType == ActivationType.RELU || activationType == ActivationType.LEAKY_RELU) {
                RandomWeights.he(weights.getValues(), totalInputs);
            } else {
                RandomWeights.xavier(weights.getValues(), totalInputs, width);
            }
        }

        biases = new float[width];
        deltaBiases = new float[width];
        prevDeltaBiases = new float[width];

        if (activationType == ActivationType.RELU || activationType == ActivationType.LEAKY_RELU) {
            Tensor.fill(biases, 0.1f);
        } else {
            Tensor.fill(biases, 0.1f);
        }

    }

    @Override
    public void forward() {
        outputs.copyFrom(biases);
        
        if (prevLayer instanceof FullyConnectedLayer || (prevLayer instanceof InputLayer && prevLayer.height == 1 && prevLayer.depth == 1)) {
            for (int outCol = 0; outCol < outputs.getCols(); outCol++) {
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                    outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));
                }
                outputs.set(outCol, activation.getValue(outputs.get(outCol)));
            }
            // outputs.apply(activation::getValue);
        } else if ((prevLayer instanceof MaxPoolingLayer) || (prevLayer instanceof ConvolutionalLayer) || (prevLayer instanceof InputLayer)) { // povezi sve na sve
            forwardFrom3DLayer();
        }

    }

    private void forwardFrom3DLayer() {
        for (int outCol = 0; outCol < outputs.getCols(); outCol++) {
            forwardFrom3DLayerForCell(outCol);
        }
    }

    private void forwardFrom3DLayerForCell(int outCol) {
        for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) {
            for (int inRow = 0; inRow < inputs.getRows(); inRow++) {
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                    outputs.add(outCol, inputs.get(inRow, inCol, inDepth) * weights.get(inCol, inRow, inDepth, outCol));
                }
            }
        }
        outputs.set(outCol, activation.getValue(outputs.get(outCol)));
    }

    @Override
    public void backward() {
        if (!batchMode) {
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        deltas.fill(0);

        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {
            for (int ndCol = 0; ndCol < nextLayer.deltas.getCols(); ndCol++) {
                deltas.add(deltaCol, nextLayer.deltas.get(ndCol) * nextLayer.weights.get(deltaCol, ndCol));
            }

            final float delta = deltas.get(deltaCol) * activation.getPrime(outputs.get(deltaCol));
            deltas.set(deltaCol, delta);
        }

        if ((prevLayer instanceof FullyConnectedLayer)
                || ((prevLayer instanceof InputLayer) && (prevLayer.height == 1 && prevLayer.depth == 1))) { 

            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { 
                for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                    final float grad = deltas.get(deltaCol) * inputs.get(inCol);
                    gradients.set(inCol, deltaCol, grad);

                    final float deltaWeight = optim.calculateDeltaWeight(grad, inCol, deltaCol);
                    deltaWeights.add(inCol, deltaCol, deltaWeight);
                }

                final float deltaBias = optim.calculateDeltaBias(deltas.get(deltaCol), deltaCol);
                deltaBiases[deltaCol] += deltaBias;
            }
        } else if ((prevLayer instanceof InputLayer)
                || (prevLayer instanceof ConvolutionalLayer)
                || (prevLayer instanceof MaxPoolingLayer)) {

            backwardTo3DLayer();
        }
    }

    private void backwardTo3DLayer() {
        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {
            backwardTo3DLayerForCell(deltaCol);
        }
    }

    private void backwardTo3DLayerForCell(int deltaCol) {

        for (int inDepth = 0; inDepth < inputs.getDepth(); inDepth++) {
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                for (int inRow = 0; inRow < inputs.getRows(); inRow++) {
                    final float grad = deltas.get(deltaCol) * inputs.get(inRow, inCol, inDepth);
                    gradients.set(inCol, inRow, inDepth, deltaCol, grad);
                    final float deltaWeight = optim.calculateDeltaWeight(grad, inCol, inRow, inDepth, deltaCol);
                    deltaWeights.add(inCol, inRow, inDepth, deltaCol, deltaWeight);
                }
            }
        }

        final float deltaBias = optim.calculateDeltaBias(deltas.get(deltaCol), deltaCol);
        deltaBiases[deltaCol] += deltaBias;
    }

    @Override
    public void applyWeightChanges() {
        if (batchMode) {
            deltaWeights.div(batchSize);
            Tensors.div(deltaBiases, batchSize);
        }

        Tensor.copy(deltaWeights, prevDeltaWeights); // save as prev delta weight
        Tensor.copy(deltaBiases, prevDeltaBiases);

        weights.add(deltaWeights);
        Tensors.add(biases, deltaBiases);

        if (batchMode) {
            deltaWeights.fill(0);
            Tensor.fill(deltaBiases, 0);
        }

    }

    @Override
    public String toString() {
        return "Fully Connected Layer { width:" + width + " activation:" + activationType.name() + "}";
    }

}
