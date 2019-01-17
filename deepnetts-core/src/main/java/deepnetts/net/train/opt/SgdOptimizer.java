package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import java.io.Serializable;

/**
 *
 * @author zoran
 */
public final class SgdOptimizer implements Optimizer, Serializable  {

    private float learningRate;
    // bias lr?
    
    public SgdOptimizer(AbstractLayer layer) {
        this.learningRate = layer.getLearningRate();
    }
    
    @Override
    public float calculateWeightDelta(final float gradient, final int... index) { // obican SGD
        return -learningRate * gradient;
    }

    @Override
    public float calculateBiasDelta(float gradient, int idx) {
        return -learningRate * gradient;
    }

    
}
