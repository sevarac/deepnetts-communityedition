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
import deepnetts.util.DeepNettsException;
import deepnetts.net.weights.RandomWeights;
import deepnetts.util.Tensor;
import java.util.logging.Logger;
import deepnetts.net.layers.activation.ActivationFunction;
import deepnetts.util.Tensors;


/**
 * Convolutional layer performs image convolution operation on outputs of a
 * previous layer using filters. This filtering operation is similar like
 * applying image filters in photoshop, but this filters can also be trained to
 * learn image features of interest.
 *
 * @author Zoran Sevarac
 */
public final class ConvolutionalLayer extends AbstractLayer {

	
	private static final long serialVersionUID = -1136065373511990910L;
	
	
    Tensor[] filters;           // each filter corresponds to a single channel. Each filter can be 3D, where 3rd dimension coreesponds to depth in previous layer. TODO: the depth pf th efilter should be tunable
    Tensor[] deltaWeights;      
    Tensor[] prevDeltaWeights;  // delta weights from previous iteration (used for momentum)
    Tensor[] prevGradSums;  // delta weights from previous iteration (used for momentum)

    /**
     * Convolutional filter width
     */
    int filterWidth,
            /**
             * Filter dimensions, filter depth is equal to number of depth /
             * depth of
             */
            filterHeight,
            /**
             * Filter dimensions, filter channels is equal to number of channels
             * / channels of
             */
            /**
             * Filter dimensions, filter depth is equal to number of depth /
             * depth of
             */
            filterDepth; // da li je filter istih dimenzija za sve feature mape?

    /**
     * Convolution step, 1 by default. Number of steps convolutional filter is
     * moved during convolution. Commonly used values 1, 2, rarely 3
     */
    int stride = 1;

    /**
     * Border padding filled with zeros (0, 1 or 2) Usually half of the filter
     * size
     */
    int padding = 0;

    int fCenterX; //  padding = (kernel-1)/2
    int fCenterY;

    int[][][][] maxIdx;


    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());

    /**
     * Create a new instance of convolutional layer with specified filter.
     * dimensions, default padding (filter-1)/2, default stride stride value 1,
     * and specified number of channels.
     *
     * @param filterWidth
     * @param filterHeight
     * @param channels
     */
    public ConvolutionalLayer(int filterWidth, int filterHeight, int channels) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = channels;
        this.stride = 1;
        this.activationType = ActivationType.TANH;
        this.activation = ActivationFunction.create(activationType);
    }

    public ConvolutionalLayer(int filterWidth, int filterHeight, int channels, ActivationType activationType) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = channels;
        this.stride = 1;
        this.activationType = activationType;
        this.activation = ActivationFunction.create(activationType);
    }

    public ConvolutionalLayer(int filterWidth, int filterHeight, int channels, int stride, ActivationType activationType) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.depth = channels;
        this.stride = stride;
        this.activationType = activationType;
        this.activation = ActivationFunction.create(activationType);
    }

    /**
     * Init dimensions, create matrices, filters, weights, biases and all
     * internal structures etc.
     *
     * Assumes that prevLayer is set in network builder
     */
    @Override
    public void init() {
        // prev layer can only be input, max pooling or convolutional
        if (!(prevLayer instanceof InputLayer || prevLayer instanceof ConvolutionalLayer || prevLayer instanceof MaxPoolingLayer)) {
            throw new DeepNettsException("Illegal architecture: convolutional layer can be used only after input, convolutional or maxpooling layer");
        }

        inputs = prevLayer.outputs;

        width = (prevLayer.getWidth()) / stride;
        height = (prevLayer.getHeight()) / stride;
        // depth is set in constructor

        fCenterX = (filterWidth - 1) / 2; //  padding = filter /2
        fCenterY = (filterHeight - 1) / 2;

        // init output cells, deltas and derivative buffer
        outputs = new Tensor(height, width, depth);
        deltas = new Tensor(height, width, depth);

        // init filters(weights)
        filterDepth = prevLayer.getDepth();
        filters = new Tensor[depth]; 
        deltaWeights = new Tensor[depth];
        prevDeltaWeights = new Tensor[depth];
        prevGradSums = new Tensor[depth];

        int inputCount = (filterWidth * filterHeight + 1) * filterDepth;

        for (int ch = 0; ch < filters.length; ch++) {
            filters[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
            RandomWeights.uniform(filters[ch].getValues(), inputCount); 
            //RandomWeights.normal(filters[ch].getValues()); 

            deltaWeights[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
            prevDeltaWeights[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
            prevGradSums[ch] = new Tensor(filterHeight, filterWidth, filterDepth);
        }

        // and biases              
        biases = new float[depth]; 
        deltaBiases = new float[depth];
        prevDeltaBiases = new float[depth];
        prevBiasSqrSum = new Tensor(depth);
        //RandomWeights.randomize(biases);        // sometimes the init to 0 for relu 0.1
        Tensor.fill(biases, 0.1f);        
    }

    /**
     * Forward pass for convolutional layer. Performs convolution operation on
     * output from previous layer using filters in this layer, on all channels.
     */
    @Override
    public void forward() {
        for (int ch = 0; ch < this.depth; ch++) {
            forwardForChannel(ch);
        }
    }

    /**
     * Performs forward pass calculation for specified channel.
     * 
     * @param ch channel to calculate
     */
    private void forwardForChannel(int ch) {
        int outRow = 0, outCol = 0;

        for (int inRow = 0; inRow < inputs.getRows(); inRow += stride) { 
            outCol = 0; 

            for (int inCol = 0; inCol < inputs.getCols(); inCol += stride) { 
                outputs.set(outRow, outCol, ch, biases[ch]); 

                for (int fz = 0; fz < filterDepth; fz++) { 
                    for (int fr = 0; fr < filterHeight; fr++) { 
                        for (int fc = 0; fc < filterWidth; fc++) { 
                            final int cr = inRow + (fr - fCenterY); 
                            final int cc = inCol + (fc - fCenterX); 

                            if (cr < 0 || cr >= inputs.getRows() || cc < 0 || cc >= inputs.getCols()) {
                                continue;
                            }

                            final float out = inputs.get(cr, cc, fz) * filters[ch].get(fr, fc, fz);
                            outputs.add(outRow, outCol, ch, out); 
                        }
                    }
                }

                // apply activation function
                final float out = activation.getValue(outputs.get(outRow, outCol, ch));
                outputs.set(outRow, outCol, ch, out);
                outCol++; 
            }
            outRow++; 
        }
    }

    /**
     * Backward pass for convolutional layer tweaks the weights in filters.
     *
     */
    @Override
    public void backward() {
        if (nextLayer instanceof FullyConnectedLayer) {
            backwardFromFullyConnected();
        }

        if (nextLayer instanceof MaxPoolingLayer) {
            backwardFromMaxPooling();
        }

        if (nextLayer instanceof ConvolutionalLayer) {
            backwardFromConvolutional();
        }
    }

    /**
     * Backward pass when next layer is fully connected.
     */
    private void backwardFromFullyConnected() {
        deltas.fill(0); 
        for (int ch = 0; ch < this.depth; ch++) {
            backwardFromFullyConnectedForChannel(ch);
        }
    }
    

    private void backwardFromFullyConnectedForChannel(int ch) {
        for (int row = 0; row < this.height; row++) {
            for (int col = 0; col < this.width; col++) {
                final float actDerivative = activation.getPrime(outputs.get(row, col, ch)); // dy/ds
                for (int ndC = 0; ndC < nextLayer.deltas.getCols(); ndC++) {
                    final float delta = nextLayer.deltas.get(ndC) * nextLayer.weights.get(col, row, ch, ndC) * actDerivative;
                    deltas.add(row, col, ch, delta);
                }
            }
        } 

        calculateDeltaWeightsForChannel(ch); 
    }  

    private void backwardFromMaxPooling() {
        final MaxPoolingLayer nextPoolLayer = (MaxPoolingLayer) nextLayer;
        maxIdx = nextPoolLayer.maxIdx; 
        deltas.fill(0);

        for (int ch = 0; ch < this.depth; ch++) {
            backwardFromMaxPoolingForChannel(ch);
        }
    }


    /**
     * Performs backward pass from maxpooling layer for specified channel in this layer.
     * 
     * @param ch 
     */
    private void backwardFromMaxPoolingForChannel(int ch) {
        for (int dr = 0; dr < nextLayer.deltas.getRows(); dr++) { 
            for (int dc = 0; dc < nextLayer.deltas.getCols(); dc++) { 

                final float nextLayerDelta = nextLayer.deltas.get(dr, dc, ch); 
                final int maxR = maxIdx[ch][dr][dc][0];
                final int maxC = maxIdx[ch][dr][dc][1];

                final float derivative = activation.getPrime(outputs.get(maxR, maxC, ch));
                deltas.set(maxR, maxC, ch, nextLayerDelta * derivative);
            }
        } 

        calculateDeltaWeightsForChannel(ch);
    }

    private void backwardFromConvolutional() {
        deltas.fill(0);

        for (int ch = 0; ch < this.depth; ch++) {
            backwardFromConvolutionalForChannel(ch);
        }
    }
    
    private void backwardFromConvolutionalForChannel(int fz) {
        ConvolutionalLayer nextConvLayer = (ConvolutionalLayer) nextLayer;  
        int filterCenterX = (nextConvLayer.filterWidth - 1) / 2;
        int filterCenterY = (nextConvLayer.filterHeight - 1) / 2;
        
        for (int ndZ = 0; ndZ < nextLayer.deltas.getDepth(); ndZ++) {
            for (int ndRow = 0; ndRow < nextLayer.deltas.getRows(); ndRow++) {
                for (int ndCol = 0; ndCol < nextLayer.deltas.getCols(); ndCol++) { 
                    final float nextLayerDelta = nextLayer.deltas.get(ndRow, ndCol, ndZ); 

                        for (int fr = 0; fr < nextConvLayer.filterHeight; fr++) {
                            for (int fc = 0; fc < nextConvLayer.filterWidth; fc++) {
                                final int row = ndRow * nextConvLayer.stride + (fr - filterCenterY); 
                                final int col = ndCol * nextConvLayer.stride + (fc - filterCenterX);

                                if (row < 0 || row >= outputs.getRows() || col < 0 || col >= outputs.getCols()) {
                                    continue;
                                }

                                final float derivative = activation.getPrime(outputs.get(row, col, fz)); 
                                deltas.add(row, col, fz, nextLayerDelta * nextConvLayer.filters[ndZ].get(fr, fc, fz) * derivative);
                            }
                        }
                   deltas.div(nextConvLayer.filterWidth*nextConvLayer.filterHeight*nextConvLayer.filterDepth);
                }
            }
        }    
        
        calculateDeltaWeightsForChannel(fz);
    }    

    
    /**
     * Calculates delta weights for the specified channel ch in this
     * convolutional layer.
     *
     * @param ch channel/depth index
     */
    private void calculateDeltaWeightsForChannel(int ch) {
        if (!batchMode) {
            deltaWeights[ch].fill(0); 
            deltaBiases[ch] = 0;
        }

         final float divisor = width * height; 

        // assumes that deltas from the next layer are allready propagated
        // calculate weight changes in filters
        for (int deltaRow = 0; deltaRow < deltas.getRows(); deltaRow++) {
            for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {
                // iterate all weights in filter for filter depth
                for (int fz = 0; fz < filterDepth; fz++) { // filter depth, input channel
                    for (int fr = 0; fr < filterHeight; fr++) {
                        for (int fc = 0; fc < filterWidth; fc++) {

                            final int inRow = deltaRow * stride + fr - fCenterY;
                            final int inCol = deltaCol * stride + fc - fCenterX;

                            if (inRow < 0 || inRow >= inputs.getRows() || inCol < 0 || inCol >= inputs.getCols()) {
                                continue;
                            }

                            final float input = inputs.get(inRow, inCol, fz); // get input for this output and weight; padding?  da li ovde imam kanal?
                            final float grad = deltas.get(deltaRow, deltaCol, ch) * input;
                            
                            
                            float deltaWeight = 0;  
                            switch (optimizerType) {
                                case SGD:
                                      deltaWeight = optim.calculateDeltaWeight(grad);
                                    break;
                                default:
                                    throw new DeepNettsException("Optimizer not supported!");
                            }
                            deltaWeight /= divisor;  // da li je ovo matematicki tacno? momentum baca nana ako ovog nema
                            deltaWeights[ch].add(fr, fc, fz, deltaWeight);
                        }
                    }
                }
                float deltaBias = 0;
                switch (optimizerType) {
                    case SGD:
                         deltaBias = optim.calculateDeltaBias(deltas.get(deltaRow, deltaCol, ch), deltaCol);
                        break;
                    default:
                         throw new DeepNettsException("Optimizer not supported!");
                }
                deltaBiases[ch] /= divisor;
                deltaBiases[ch] += deltaBias;
            }
        }
    }

    /**
     * Apply weight changes calculated in backward pass
     */
    @Override
    public void applyWeightChanges() {

        if (batchMode) {
            Tensors.div(deltaBiases, batchSize);
        }

        Tensor.copy(deltaBiases, prevDeltaBiases);  // save this for momentum

        for (int ch = 0; ch < depth; ch++) {
            if (batchMode) { 
                deltaWeights[ch].div(batchSize);
            }

            Tensor.copy(deltaWeights[ch], prevDeltaWeights[ch]); 

            filters[ch].add(deltaWeights[ch]);
            biases[ch] += deltaBiases[ch];

            if (batchMode) {    // reset delta weights for next batch
                deltaWeights[ch].fill(0);
            }
        }

        if (batchMode) { // reset delta biases for next batch
            Tensor.fill(deltaBiases, 0);
        }

    }

    public Tensor[] getFilters() {
        return filters;
    }

    public void setFilters(Tensor[] filters) {
        this.filters = filters;
    }

    public void setFilters(String filtersStr) {

        String[] strVals = filtersStr.split(";"); 
        int filterSize = filterWidth * filterHeight * filterDepth;

        for (int i = 0; i < filters.length; i++) {
            float[] filterValues = new float[filterSize];
            String[] vals = strVals[i].split(",");
            for (int k = 0; k < filterSize; k++) {
                filterValues[k] = Float.parseFloat(vals[k]);
            }

            filters[i].setValues(filterValues); 
        }
    }

    public int getFilterWidth() {
        return filterWidth;
    }

    public int getFilterHeight() {
        return filterHeight;
    }

    public int getFilterDepth() {
        return filterDepth;
    }

    public int getStride() {
        return stride;
    }

    public Tensor[] getFilterDeltaWeights() {
        return deltaWeights;
    }

    @Override
    public String toString() {
        return "Convolutional Layer { filter width:" + filterWidth + ", filter height: " + filterHeight + ", channels: " + depth + ", stride: " + stride + ", activation: " + activationType.name() + "}";
    }

}
