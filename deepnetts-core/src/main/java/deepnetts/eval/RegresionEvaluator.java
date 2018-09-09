package deepnetts.eval;

import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.NeuralNetwork;

/**
 *
 * @author Zoran
 */
public class RegresionEvaluator implements Evaluator<NeuralNetwork, DataSet<?>> {

    @Override
    public PerformanceMeasure evaluatePerformance(NeuralNetwork neuralNet, DataSet<?> testSet) {
        PerformanceMeasure pe = new PerformanceMeasure();

        MeanSquaredError mse = new MeanSquaredError();

        for (DataSetItem item : testSet) {
            neuralNet.setInput(item.getInput());
            neuralNet.forward();
            mse.add(neuralNet.getOutput(), item.getTargetOutput());
        }

        pe.set(PerformanceMeasure.MSE, mse.getTotal());

        return pe;
    }

}
