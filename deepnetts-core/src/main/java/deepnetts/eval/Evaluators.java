package deepnetts.eval;

import deepnetts.data.DataSet;
import deepnetts.net.NeuralNetwork;

/**
 * This class provides various utility methods for evaluating machine learning models.
 * @author Zoran Sevarac
 */
public class Evaluators {

    private Evaluators() { }

    /**
     *
     * @param neuralNet
     * @param testSet
     * @return regression performance measures
     */
    public static PerformanceMeasure evaluateRegressor(NeuralNetwork<?> neuralNet, DataSet<?> testSet) {
        RegresionEvaluator eval = new RegresionEvaluator();
        return eval.evaluatePerformance(neuralNet, testSet);
    }

    /**
     *
     * @param neuralNet
     * @param testSet
     * @return classification performance measure
     */
    public static PerformanceMeasure evaluateClassifier(NeuralNetwork<?> neuralNet, DataSet<?> testSet) {
        ClassifierEvaluator eval = new ClassifierEvaluator();
        return eval.evaluatePerformance(neuralNet, testSet);
    }

}