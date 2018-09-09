package deepnetts.net.train.optimizer;

public final class SGDOptimizer extends AbstractOptimizer {

    // optimizeri treba da imaju dodatne strukture koje su im potrebne
    @Override
    public float calculate(final float gradient) {
        return -learningRate * gradient;
    }

}
