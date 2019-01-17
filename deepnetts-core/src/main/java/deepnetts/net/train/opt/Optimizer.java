package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.util.DeepNettsException;

public interface Optimizer {
    
    public float calculateWeightDelta(final float gradient, final int... index);
    public float calculateBiasDelta(final float gradient, final int idx);
    
    //alternatije je da imam rows, cols i sa 4 indeksa to je hardkoriano samo za ovaj lib!!!
    
   // public float calculateWeightDelta(final float gradient, final int row, final int col);
    //public float calculateWeightDelta(final float gradient, final int row, final int col, final int depth, final int fourth);
    // public float calculateWeightDelta(final float gradient, final int col); za biase
        
    public static Optimizer create(OptimizerType type, AbstractLayer layer) {
        switch (type) {
            case SGD:
                return new SgdOptimizer(layer);
            case MOMENTUM:
                return new MomentumOptimizer(layer);
            default:
                throw new DeepNettsException("Unknown optimizer:" + type);
        }
    }    
    
}
