package deepnetts.eval;

import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.NeuralNetwork;

public class RegresionEvaluator implements Evaluator<NeuralNetwork, DataSet<?>> {

    @Override
    public EvaluationMetrics evaluate(NeuralNetwork neuralNet, DataSet<?> testSet) {
        EvaluationMetrics pe = new EvaluationMetrics();

        MeanSquaredError mse = new MeanSquaredError();

        for (DataSetItem item : testSet) {
            neuralNet.setInput(item.getInput());
            neuralNet.forward();
            mse.add(neuralNet.getOutput(), item.getTargetOutput());
        }

        final float rss = mse.getSquaredSum();

        float rse = (float)Math.sqrt(rss / (float)(testSet.size() - 2));
        pe.set(EvaluationMetrics.RESIDUAL_STANDARD_ERROR, rse);

        // tss squared sum of diffes between mean value and predicted
        //float r2 = 1- rss/tss;
        pe.set(EvaluationMetrics.R2, Float.NaN);

        pe.set(EvaluationMetrics.MEAN_SQUARED_ERROR, mse.getMeanSquaredSum());  // this is used for training

        return pe;
    }

}
