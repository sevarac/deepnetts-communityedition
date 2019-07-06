package deepnetts.eval;

import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.NeuralNetwork;

/**
 * Evaluates regressor neural network for specified data set.
 * Assumes only one output at the moment.
 * 
 * @author Zoran Sevarac
 */
public class RegresionEvaluator implements Evaluator<NeuralNetwork, DataSet<?>> {

    @Override
    public EvaluationMetrics evaluate(NeuralNetwork neuralNet, DataSet<?> testSet) {
        EvaluationMetrics pe = new EvaluationMetrics();

        MeanSquaredError mse = new MeanSquaredError();
        float tss=0;
        int numInputs = testSet.get(0).getInput().size();
        int numItems= testSet.size();
        float targetMean = mean(testSet);

        for (DataSetItem item : testSet) {
            neuralNet.setInput(item.getInput());
            float[] predicted = neuralNet.getOutput();
            mse.add(predicted, item.getTargetOutput().getValues());
            tss += (item.getTargetOutput().getValues()[0] - targetMean)*(item.getTargetOutput().getValues()[0] - targetMean);
        }

        final float rss = mse.getSquaredSum();

        float rse = (float)Math.sqrt(rss / (float)(testSet.size() - 2));
        pe.set(EvaluationMetrics.RESIDUAL_STANDARD_ERROR, rse);

        float r2 = 1- rss/tss;
        pe.set(EvaluationMetrics.R_SQUARED, r2);

        
        final float fStat = ((tss-rss)/(float)numInputs) / (float)(rss / ( numItems - numInputs - 1));
        pe.set(EvaluationMetrics.F_STAT, fStat);
        
        pe.set(EvaluationMetrics.MEAN_SQUARED_ERROR, mse.getMeanSquaredSum());

        return pe;
    }

    private float mean(DataSet<? extends DataSetItem> testSet) {
        float mean=0;
        for(DataSetItem ditem : testSet) {
            mean += ditem.getTargetOutput().get(0);
        }
        return mean / testSet.size();
    }

}
