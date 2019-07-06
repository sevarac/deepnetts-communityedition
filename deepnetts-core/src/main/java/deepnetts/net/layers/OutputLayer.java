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
//import deepnetts.net.train.opt.Optimizer;
//import deepnetts.net.train.opt.OptimizerType;
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

    protected float[] outputErrors;
    protected final String[] labels;
    protected LossType lossType;
  //  protected Optimizer optim;

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

//    @Override
//    public void setOptimizerType(OptimizerType optimizer) {
//        super.setOptimizerType(optimizer);
//        optim = Optimizer.create(optimizer, this);
//    }

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

//        setOptimizerType(OptimizerType.SGD);
    }

    /**
     * This method implements forward pass for the output layer.
     *
     * Calculates weighted input and layer outputs using sigmoid function.
     */
    @Override
    public void forward() {
        outputs.copyFrom(biases);  // reset output to bias value, so weighted sum is added to biases
        // mnozenje vektora (input) matricom (weights) i smestanje u outputs vector
        for (int outCol = 0; outCol < outputs.getCols(); outCol++) {  // for all neurons in this layer   ForkJoin split this in two until you reach size which makes sense: number of calculations = inputCols * outputCols
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                outputs.add(outCol, inputs.get(inCol) * weights.get(inCol, outCol));    // add weighted sum
            }
        }

        outputs.apply(activation::getValue);    // apply activation function to all outputs
    }

    /**
     * This method implements backward pass for the output layer.
     *
     * http://peterroelants.github.io/posts/neural_network_implementation_intermezzo01/
     * http://neuralnetworksanddeeplearning.com/chap3.html#introducing_the_cross-entropy_cost_function
     */
    @Override
    public void backward() {
        if (!batchMode) {   // reset delta weights and deltaBiases to zero in each iteration if not in batch/minibatch mode
            deltaWeights.fill(0);
            Arrays.fill(deltaBiases, 0);
        }

        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) { // iterate all output neurons / deltas
            if (lossType == LossType.MEAN_SQUARED_ERROR) {
                final float delta = outputErrors[deltaCol] * activation.getPrime(outputs.get(deltaCol)); // delta = e*dE/ds
                deltas.set(deltaCol, delta);
            } else if (activationType == ActivationType.SIGMOID && lossType == LossType.CROSS_ENTROPY) { // ovo samo za binary cross entropy, single sigmoid output
                deltas.set(deltaCol, outputErrors[deltaCol]); // Bishop, pg. 231, eq.6.125, imenilac od dE/dy i izvod sigmoidne se skrate
            } // ... slucaj Cross entropy sa softmax je resen u SoftMaxLayer

            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
               final float grad = deltas.get(deltaCol) * inputs.get(inCol);
             //   final float grad = deltas.get(deltaCol) * inputs.get(inCol) + 2 * regularization * weights.get(inCol, deltaCol); // gradient dE/dw + regularization l2
//                final float grad = deltas.get(deltaCol) * inputs.get(inCol) + 0.01f * ( weights.get(inCol, deltaCol)>=0? 1 : -1 ); // gradient dE/dw + regularization l2

                gradients.set(inCol, deltaCol, grad);

                //final float deltaWeight = Optimizers.sgd(learningRate, grad);
                //final float deltaWeight = Optimizers.momentum(learningRate, grad, momentum, prevDeltaWeights.get(inCol, dCol));

                final float deltaWeight = optim.calculateDeltaWeight(grad, inCol, deltaCol);
                deltaWeights.add(inCol, deltaCol, deltaWeight); // sum deltaWeight for batch mode
            }

            final float deltaBias = optim.calculateDeltaBias(deltas.get(deltaCol), deltaCol);
            deltaBiases[deltaCol] += deltaBias; //Optimizers.sgd(learningRate, deltas.get(deltaCol));
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
            Tensors.div(deltaBiases, batchSize);
        }

        // save current as prev delta weights (required for momentum)
        Tensor.copy(deltaWeights, prevDeltaWeights);
        // apply(add) delta weights
        weights.add(deltaWeights);

        // save current as prev delta biases
        Tensor.copy(deltaBiases, prevDeltaBiases);
        // apply(add) delta bias
        Tensors.add(biases, deltaBiases);

        if (batchMode) {    // for batch mode set all delta weights and biases to zero after applying changes. For online mode they are reseted in backward pass
            deltaWeights.fill(0);
            Tensor.fill(deltaBiases, 0);
        }
    }
    
    @Override
    public String toString() {
        return "Output Layer { width:"+width+", activation:"+activationType.name()+"}";
    }    
}
