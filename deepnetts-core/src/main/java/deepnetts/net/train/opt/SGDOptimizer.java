package deepnetts.net.train.opt;

import java.io.Serializable;

public final class SGDOptimizer extends AbstractOptimizer implements Serializable {

    // optimizeri treba da imaju dodatne strukture koje su im potrebne
    @Override
    public float calculate(final float gradient) {
        return -learningRate * gradient;
    }

}
