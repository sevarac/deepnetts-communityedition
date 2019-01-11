package deepnetts.eval;

import deepnetts.data.DataSet;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;

/**
 *
 * @author zoran
 */
public class Evaluators {

    public static PerformanceMeasure evaluateRegressor(NeuralNetwork<?> neuralNet, DataSet<?> testSet) {
        RegresionEvaluator eval = new RegresionEvaluator();
        return eval.evaluatePerformance(neuralNet, testSet);
    }
    
    private Evaluators() { }
    
    public static PerformanceMeasure evaluateClassifier(NeuralNetwork<?> neuralNet, DataSet<?> testSet) {
        ClassifierEvaluator eval = new ClassifierEvaluator();
        return eval.evaluatePerformance(neuralNet, testSet);
    }
    
}
