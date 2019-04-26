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
    
        // index is not used!
    @Override
    public float calculateDeltaWeight(final float gradient, final int... index) { // obican SGD
        return -learningRate * gradient;
    }

    @Override
    public float calculateDeltaBias(float gradient, int idx) {
        return -learningRate * gradient;
    }

    
}
