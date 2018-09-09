package deepnetts.net.train.optimizer;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.util.Tensor;

public abstract class AbstractOptimizer implements Optimizer {

    protected Tensor deltas;
    protected Tensor inputs;
    protected Tensor gradients;
    protected float learningRate;   // this can also be for each weight!
    protected Tensor deltaWeights;
    protected float[] deltaBiases;

    // ovo treba napraviti tako da na istu osnovu mogu lako da se dodaju optimizeri
    @Override
    public void optimize(AbstractLayer layer) {
        deltas = layer.getDeltas();
        gradients = layer.getGradients();
        deltaWeights = layer.getDeltaWeight();
        deltaBiases = layer.getBiases();
        inputs = layer.getPrevlayer().getOutputs();
        learningRate = layer.getLearningRate();

        for (int dCol = 0; dCol < deltas.getCols(); dCol++) { // this iterates neurons
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {
                final float grad = deltas.get(dCol) * inputs.get(inCol); // gradient dE/dw
                gradients.set(inCol, dCol, grad);

                final float deltaWeight = calculate(grad); // zapravo ne bih ja ovo slao dalje u petlji, nece da inlinuje...
                deltaWeights.add(inCol, dCol, deltaWeight);
            }

            final float deltaBias = calculate(deltas.get(dCol));
            deltaBiases[dCol] += deltaBias;
        }
    }

    // this method implemenst specific formulas in subclasses
    public abstract float calculate(final float grad);

}
