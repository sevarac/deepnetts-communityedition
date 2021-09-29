/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */

package deepnetts.net.layers;

import deepnetts.core.DeepNetts;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.Tensor;
import java.util.concurrent.Callable;
import java.util.concurrent.CyclicBarrier;
import java.util.logging.Logger;

/**
 * This layer performs max pooling operation in convolutional neural network, which
 * scales down output from previous layer by taking max outputs from small predefined filter areas.
 *
 * @see ConvolutionalNetwork
 * @author Zoran Sevarac
 */
public final class MaxPoolingLayer extends AbstractLayer {

	
	private static final long serialVersionUID = -5187033033631855969L;
	
	
    /**
     * Filter dimensions.
     *
     * Commonly used 2x2 with stride 2
     */
    final int filterWidth, filterHeight;

    /**
     * Filter step.
     *
     * Commonly used 2
     */
    final int stride;

    /**
     * Max activation idxs.
     *
     * Remember idx of max output for each filter position. [channel][row][col][2]
     */
    int maxIdx[][][][];

    private static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());
    
    /**
     * Creates a new max pooling layer with specified filter dimensions and stride.
     *
     * @param filterWidth width of the filter square
     * @param filterHeight height of the filter square
     * @param stride filter step
     */
    public MaxPoolingLayer(int filterWidth, int filterHeight, int stride) {
        this.filterWidth = filterWidth;
        this.filterHeight = filterHeight;
        this.stride = stride;
    }

    @Override
    final public void init() {
        // max pooling layer can be only after Convolutional Layer
        if (!(prevLayer instanceof ConvolutionalLayer)) throw new RuntimeException("Illegal network architecture! MaxPooling can be only after convolutional layer!");

        inputs = prevLayer.outputs;

        width = (inputs.getCols() - filterWidth) / stride + 1; 
        height = (inputs.getRows() - filterHeight) / stride + 1;
        depth = prevLayer.getDepth(); 

        outputs = new Tensor(height, width, depth);
        deltas = new Tensor(height, width,  depth);

        // used in fprop to save idx position of max value
        maxIdx = new int[depth][height][width][2];

    }


    /**
     * Max pooling forward pass outputs the max value for each filter position.
     */
    @Override
    public void forward() {
        for (int ch = 0; ch < this.depth; ch++) {  
            forwardForChannel(ch);
        }
    }

    private void forwardForChannel(int ch) {
        float max; // max value
        int maxC = -1, maxR = -1;

        int outCol = 0, outRow = 0;

        for (int inRow = 0; inRow < inputs.getRows() - filterHeight + 1; inRow += stride) {
            outCol = 0; // reset col on every new row ???????
            for (int inCol = 0; inCol < inputs.getCols() - filterWidth + 1; inCol += stride) {

                // apply max pool filter
                max = inputs.get(inRow, inCol, ch);
                maxC = inCol;
                maxR = inRow;
                for (int fr = 0; fr < filterHeight; fr++) {
                    for (int fc = 0; fc < filterWidth; fc++) {
                        if (max < inputs.get(inRow + fr, inCol + fc, ch)) {
                            maxR = inRow + fr;
                            maxC = inCol + fc;
                            max = inputs.get(maxR, maxC, ch);
                        }
                    }
                }

                maxIdx[ch][outRow][outCol][0] = maxR; // height idx (row)
                maxIdx[ch][outRow][outCol][1] = maxC; // width idx (col)

                outputs.set(outRow, outCol, ch, max); // set max value as output
                outCol++;   
            } // scan col
            outRow++; 
        } // scan row
    }

    /**
     * Backward pass for a max(x, y) operation has a simple interpretation as
     * only routing the gradient to the input that had the highest value in the
     * forward pass.
     *
     */
    @Override
    public void backward() {

        if (nextLayer instanceof FullyConnectedLayer) {
            backwardFromFullyConnected();
        }
        else if (nextLayer instanceof ConvolutionalLayer) {
             backwardFromConvolutional();
        }

    }

    private void backwardFromFullyConnected() {
        deltas.fill(0); 

        for (int ch = 0; ch < deltas.getDepth(); ch++) {
            backwardFromFullyConnectedForChannel(ch);
        }
    }
    
    private void backwardFromFullyConnectedForChannel(int ch) {
        for (int row = 0; row < deltas.getRows(); row++) {
            for (int col = 0; col < deltas.getCols(); col++) {
                for (int ndC = 0; ndC < nextLayer.deltas.getCols(); ndC++) {
                    final float nextLayerDelta = nextLayer.deltas.get(ndC);
                    final float weight = nextLayer.weights.get(col, row, ch, ndC);
                    deltas.add(row, col, ch, nextLayerDelta * weight);
                }
            }
        }
    }
    
    private void backwardFromConvolutional() {
        deltas.fill(0);

        for (int ch = 0; ch < depth; ch++) {
            backwardFromConvolutionalForChannel(ch);
        }
    }
    
    private void backwardFromConvolutionalForChannel(int fz) {
        final ConvolutionalLayer nextConvLayer = (ConvolutionalLayer) nextLayer;
        final int filterCenterX = (nextConvLayer.filterWidth - 1) / 2;
        final int filterCenterY = (nextConvLayer.filterHeight - 1) / 2;

        // 1. Propagate deltas from next conv layer for max outputs from this layer
        for (int ndz = 0; ndz < nextLayer.deltas.getDepth(); ndz++) { 
            for (int ndr = 0; ndr < nextLayer.deltas.getRows(); ndr++) { 
                for (int ndc = 0; ndc < nextLayer.deltas.getCols(); ndc++) { 
                    final float nextLayerDelta = nextLayer.deltas.get(ndr, ndc, ndz); 

                    for (int fr = 0; fr < nextConvLayer.filterHeight; fr++) {
                        for (int fc = 0; fc < nextConvLayer.filterWidth; fc++) {
                            final int outRow = ndr * nextConvLayer.stride + (fr - filterCenterY);
                            final int outCol = ndc * nextConvLayer.stride + (fc - filterCenterX);

                            if (outRow < 0 || outRow >= outputs.getRows() || outCol < 0 || outCol >= outputs.getCols()) {
                                continue;
                            }

                            deltas.add(outRow, outCol, fz, nextLayerDelta * nextConvLayer.filters[ndz].get(fr, fc, fz)); // da li se ovde preo z preklapaju?
                        }
                    }
                }
            }
        }
    }


    /**
     * Does nothing for pooling layer since it does not have weights
     * It just propagates deltas from next layer to previous through connections that had max activation in forward pass
     */
    @Override
    public void applyWeightChanges() {    }

    public int getFilterWidth() {
        return filterWidth;
    }

    public int getFilterHeight() {
        return filterHeight;
    }

    public int getStride() {
        return stride;
    }
    
    @Override
    public String toString() {
        return "Max Pooling Layer { filter width:"+filterWidth+", filter height: "+filterHeight+", stride:"+stride+"}";
    }

}