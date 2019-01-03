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

import deepnetts.net.layers.activation.ActivationFunctions;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.Optimizers;
import deepnetts.util.WeightsInit;
import deepnetts.util.Tensor;
import java.util.Arrays;
import deepnetts.net.layers.activation.ActivationFunction;

/**
 * This class represents output layer with sigmoid output function by default.
 *
 * @author zoran
 */
public class OutputLayer extends AbstractLayer {

    protected float[] outputErrors;
    protected final String[] labels;
    protected LossType lossType;

    public OutputLayer(int width) {
        this.width = width;
        this.height = 1;
        this.depth = 1;

        labels = new String[depth];
        // generate enumerated class names from 1..n
        for (int i = 0; i < depth; i++) {
            labels[i] = "Output" + i;
        }

        setActivationType(ActivationType.SIGMOID);
        this.activation = ActivationFunction.create(ActivationType.SIGMOID);
    }

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
        this.activation = ActivationFunction.create(actType);
    }

    public OutputLayer(String[] labels) {
        this.width = labels.length;
        this.height = 1;
        this.depth = 1;
        this.labels = labels;
        setActivationType(ActivationType.SIGMOID);
        this.activation = ActivationFunction.create(ActivationType.SIGMOID);
    }

    public OutputLayer(String[] labels, ActivationType activationFunction) {
        this(labels);
        setActivationType(activationFunction);
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
        WeightsInit.xavier(weights.getValues(), prevLayerWidth, width);

        biases = new float[width];
        deltaBiases = new float[width];
        prevDeltaBiases = new float[width];
        WeightsInit.randomize(biases);
    }

    /**
     * This method implements forward pass for the output layer.
     *
     * Calculates weighted input and layer outputs using sigmoid function.
     */
    @Override
    public void forward() {
        outputs.copyFrom(biases);  // reset output to bias value, so weighted sum is added to biases
        for (int outCol = 0; outCol < outputs.getCols(); outCol++) {  // for all neurons in this layer   ForkJoin split this in two until you reach size which makes sense: number of calculations = inputCols * outputCols
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));    // add weighted sum
            }
           // outputs.set(outCol, ActivationFunctions.calc(activationType, outputs.get(outCol)));
        }
        outputs.apply(activation::getValue);
    }

    /**
     * This method implements backward pass for the output layer.
     *
     * http://peterroelants.github.io/posts/neural_network_implementation_intermezzo01/
     * http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
     */
    @Override
    public void backward() {
        if (!batchMode) {   // reset delta weights and deltaBiases to zero in ezch iteration if not in batch/minibatch mode
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { // iterate all output neurons / deltas

            if (lossType == LossType.MEAN_SQUARED_ERROR) {
                deltas.set(deltaCol, outputErrors[deltaCol] * ActivationFunctions.prime(activationType, outputs.get(deltaCol))); // delta = e*f1
            } else if (activationType == ActivationType.SIGMOID && lossType == LossType.CROSS_ENTROPY) { // ovo samo za binary cross entropy, single sigmoid output
                deltas.set(deltaCol, outputErrors[deltaCol]); // Bishop, pg. 231, eq.6.125, imenilac od dE/dy i izvod sigmoidne se skrate
            }

            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
               // final float grad = deltas.get(dCol) * inputs.get(inCol);
                final float grad = deltas.get(deltaCol) * inputs.get(inCol) + 2 * regularization * weights.get(inCol, deltaCol); // gradient dE/dw + regularization l2
//                final float grad = deltas.get(deltaCol) * inputs.get(inCol) + 0.01f * ( weights.get(inCol, deltaCol)>=0? 1 : -1 ); // gradient dE/dw + regularization l2
                 
                final float deltaWeight = Optimizers.sgd(learningRate, grad);
                //final float deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, dCol));
                gradients.set(inCol, deltaCol, grad);
                deltaWeights.add(inCol, deltaCol, deltaWeight); // sum deltaWeight for batch mode
            }

            deltaBiases[deltaCol] += Optimizers.sgd(learningRate, deltas.get(deltaCol));
//          deltaBiases[dCol] += Optimizers.momentum(learningRate, deltas.get(dCol), momentum, prevDeltaBiases[dCol]);
        }
    }

    /**
     * Applies weight changes after one learning iteration or batch
     */
    @Override
    public void applyWeightChanges() {
        if (batchMode) { // if batch mode calculate average delta weights using batch samples (mini batch)
            deltaWeights.div(batchSize);
            Tensor.div(deltaBiases, batchSize);
        }

        // save current as prev delta weights (required for momentum)
        Tensor.copy(deltaWeights, prevDeltaWeights);
        // apply(add) delta weights
        weights.add(deltaWeights);

        // save current as prev delta biases
        Tensor.copy(deltaBiases, prevDeltaBiases);
        // apply(add) delta bias
        Tensor.add(biases, deltaBiases);

        if (batchMode) {    // for batch mode set all delta weights and biases to zero after applying changes. For online mode they are reseted in backward pass
            deltaWeights.fill(0);
            Tensor.fill(deltaBiases, 0);
        }
    }

}
