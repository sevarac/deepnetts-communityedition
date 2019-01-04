package deepnetts.eval;

import deepnetts.data.DataSet;
import deepnetts.net.NeuralNetwork;

/**
 *
 * @author zoran
 */
public class Evaluators {
    
    private Evaluators() { }
    
    public static PerformanceMeasure evaluateClassifier(NeuralNetwork<?> neuralNet, DataSet<?> testSet) {
        ClassifierEvaluator eval = new ClassifierEvaluator();
        return eval.evaluatePerformance(neuralNet, testSet);
    }
    
}
