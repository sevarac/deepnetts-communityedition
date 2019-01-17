package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.util.Tensor;
import java.io.Serializable;

public final class MomentumOptimizer implements Optimizer, Serializable {

    public static int ROW_IDX=0;
    public static int COL_IDX=1;
    
    private float momentum;
    private float learningRate;        
    private final Tensor prevDeltaWeights;
    private final float[] prevDeltaBiases;
    
    public MomentumOptimizer(AbstractLayer layer) {
        this.learningRate = layer.getLearningRate();
        this.momentum = layer.getMomentum();
        this.prevDeltaWeights = layer.getPrevDeltaWeights();
        this.prevDeltaBiases = layer.getPrevDeltaBiases();
    }
    
    @Override
    public float calculateWeightDelta(final float grad, final int... idxs) { // momentum with idxs
        return -learningRate * grad + momentum * prevDeltaWeights.get(idxs[ROW_IDX], idxs[COL_IDX]);
    }
    
   public float calculateParamDelta(final float grad, final int row, final int col, final int depth, final int fourth) {    // momentum
        return -learningRate * grad + momentum * prevDeltaWeights.get(row, col, depth, fourth);  // inCol, inRow, inDepth, deltaCol
    }    
    

  //  @Override
    public float calculateParamDelta(final float grad, final int row, final int col) {    // momentum
        return -learningRate * grad + momentum * prevDeltaWeights.get(row, col); 
    }

 //   @Override
    public float calculateParamDelta(final float grad, final int col) {  // momentum for biases
        return -learningRate * grad + momentum * prevDeltaWeights.get(col);  // ovde prev biases
    }

 //   @Override
    public float calculateParamDelta(final float grad) { // obican SGD
        return -learningRate * grad;
    }

    @Override
    public float calculateBiasDelta(float gradient, int idx) {
         return -learningRate * gradient + momentum * prevDeltaBiases[idx];  // ovde prev biases
    }

}