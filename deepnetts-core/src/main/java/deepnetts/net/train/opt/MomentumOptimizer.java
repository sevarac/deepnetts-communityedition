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

package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.util.DeepNettsException;
import deepnetts.util.Tensor;
import java.io.Serializable;

public final class MomentumOptimizer implements Optimizer, Serializable {

    public final static int ROW_IDX=0;
    public final static int COL_IDX=1;
    
    private float momentum;
    private float learningRate;        
    private final Tensor prevDeltaWeights;  // ove moraju da budu posebne za svaki kanal u conv layeru su u nizu mora da bude niz tensora ili tensor da ima jos jednu dimenziju!!!
    private final float[] prevDeltaBiases;
    
    AbstractLayer layer;
    
    public MomentumOptimizer(AbstractLayer layer) {
        this.layer = layer;
        this.learningRate = layer.getLearningRate();
        this.momentum = layer.getMomentum();
        this.prevDeltaWeights = layer.getPrevDeltaWeights();
        this.prevDeltaBiases = layer.getPrevDeltaBiases();
    }
    
    @Override
    public float calculateDeltaWeight(final float grad, final int... idxs) { // momentum with idxs
        if (idxs.length==2)
            return -learningRate * grad + momentum * prevDeltaWeights.get(idxs[ROW_IDX], idxs[COL_IDX]); 
        else if (idxs.length==4) {
            final float dw = -learningRate * grad + momentum * prevDeltaWeights.get(idxs[ROW_IDX], idxs[COL_IDX], idxs[2], idxs[3]/*, inDepth, deltaCol*/);    // ch? z?
            if (dw == Float.NaN) {
                throw new DeepNettsException("NaN in momentum!!!"+layer.getClass());
            }
            return dw;
        }
        else
            return -learningRate * grad + momentum * prevDeltaWeights.get(idxs[ROW_IDX], idxs[COL_IDX]); 
        //  -learningRate * gradient + momentum * prevDeltaWeight;
        // ovako se poziva:  optim.calculateDeltaWeight(grad, inCol, inRow, inDepth, deltaCol); depth i delta col dimenzije se ni ne koriste!!!!!!!
                                                           // inCol, inRow, inDepth, deltaCol        
    }
//    
//   public float calculateParamDelta(final float grad, final int row, final int col, final int depth, final int fourth) {    // momentum
//        return -learningRate * grad + momentum * prevDeltaWeights.get(row, col, depth, fourth);  // inCol, inRow, inDepth, deltaCol
//    }    
//    
//
//  //  @Override
//    public float calculateParamDelta(final float grad, final int row, final int col) {    // momentum
//        return -learningRate * grad + momentum * prevDeltaWeights.get(row, col); 
//    }
//
// //   @Override
//    public float calculateParamDelta(final float grad, final int col) {  // momentum for biases
//        return -learningRate * grad + momentum * prevDeltaWeights.get(col);  // ovde prev biases
//    }
//
// //   @Override
//    public float calculateParamDelta(final float grad) { // obican SGD
//        return -learningRate * grad;
//    }

    @Override
    public float calculateDeltaBias(float gradient, int idx) {
         return -learningRate * gradient + momentum * prevDeltaBiases[idx];  // ovde prev biases
    }

}