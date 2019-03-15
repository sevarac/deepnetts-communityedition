/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */
package deepnetts.eval;

import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.NeuralNetwork;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Evaluation method for binary and multi-class classifiers.
 * Calculates classification performance metrics, how good is it at predicting class of something.
 *
 * http://www.ritchieng.com/machine-learning-evaluate-classification-model/
 * http://scikit-learn.org/stable/modules/model_evaluation.html
 * http://notesbyanerd.com/2014/12/17/multi-class-performance-measures/
 * https://en.wikipedia.org/wiki/Confusion_matrix
 * https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
 * http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
 *
 * Micro and macro averaging: https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
 * I'm doing macro
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */                                         // Evaluator<Classifier, AbstractClassifier, annotate NeuralNetwork instance to become a Classifier
public class ClassifierEvaluator implements Evaluator<NeuralNetwork, DataSet<?>> { // use Classifier as a generic, wrap convolutional network with classifier

    /**
     * Constants used as labels for binary classification
     */
    private final static String POSITIVE = "positive";
    private final static String NEGATIVE = "negative";

    /**
     * Class labels
     */
    private List<String> classLabels = new ArrayList<>();

    /**
     * Confusion matrix that holds classification results
     */
    private ConfusionMatrix confusionMatrix;

    /**
     * Performance measures for each class, used for multi class classification
     */
    private HashMap<String, PerformanceMeasure> performanceByClass;

    private float threshold = 0.5f;

    private void init() {
        performanceByClass = new HashMap<>();

        if (classLabels.size() == 1) {
            confusionMatrix = new ConfusionMatrix(new String[]{POSITIVE, NEGATIVE}); // labels for binary classification
        } else {
            confusionMatrix = new ConfusionMatrix(classLabels.toArray(new String[classLabels.size()]));
            classLabels.forEach((label) -> {
                performanceByClass.put(label, new PerformanceMeasure());
            });
        }
    }

    @Override
    public PerformanceMeasure evaluatePerformance(NeuralNetwork neuralNet, DataSet<?> testSet) {
        classLabels = Arrays.asList(testSet.getOutputLabels()); // get labels from neural network outputs

        // if class labels are empty create class1, class2, classk ....
        init();

        //  wrap neural network with classifier interface, setInput from param, and return output
        // I need a method that wraps modelinto a classifier Classifier.fromNeuralNetwork(neuralNet)
        // so this method can accept classifier as aparam
        for (DataSetItem item : testSet) {
            neuralNet.setInput(item.getInput());
            final float[] predictedOut = neuralNet.getOutput();
            processResult(item.getTargetOutput(), predictedOut);
        }

        if (classLabels.size() == 1) {  // for binary classification
            return createBinaryPerformanceMeasures();
        } else {  // for multi class classification
            createMultiClassPerformanceMeasures();
            return getTotalAverage(); // ovde vracam total posebnom metodom uzimam by class
        }
    }

    private PerformanceMeasure createBinaryPerformanceMeasures() {
        PerformanceMeasure pm = new PerformanceMeasure();

        int tp = confusionMatrix.getTruePositive();
        int tn = confusionMatrix.getTrueNegative();
        int fp = confusionMatrix.getFalsePositive();
        int fn = confusionMatrix.getFalseNegative();

        ClassificationMetrics cm = new ClassificationMetrics(tn, fp, fn, tp);

        pm.set("TotalClasses", classLabels.size());
        pm.set("TotalItems", cm.getTotal());

        pm.set(ConfusionMatrix.TRUE_POSITIVE, tp);
        pm.set(ConfusionMatrix.TRUE_NEGATIVE, tn);
        pm.set(ConfusionMatrix.FALSE_POSITIVE, fp);
        pm.set(ConfusionMatrix.FALSE_NEGATIVE, fn);

        pm.set("TotalCorrect", tp + tn);
        pm.set("TotalIncorrect", fp + fn);

        pm.set(PerformanceMeasure.ACCURACY, cm.getAccuracy());
        pm.set(PerformanceMeasure.PRECISION, cm.getPrecision());
        pm.set(PerformanceMeasure.RECALL, cm.getRecall());
        pm.set(PerformanceMeasure.F1SCORE, cm.getF1Score());

        return pm;
    }

    private Map<String, PerformanceMeasure> createMultiClassPerformanceMeasures() {
        performanceByClass = new HashMap();
        for (int clsIdx = 0; clsIdx < classLabels.size(); clsIdx++) {
            PerformanceMeasure pm = new PerformanceMeasure();

            int tp = confusionMatrix.getTruePositive(clsIdx);
            int tn = confusionMatrix.getTrueNegative(clsIdx);
            int fp = confusionMatrix.getFalsePositive(clsIdx);
            int fn = confusionMatrix.getFalseNegative(clsIdx);

            pm.set(ConfusionMatrix.TRUE_POSITIVE, tp);
            pm.set(ConfusionMatrix.TRUE_NEGATIVE, tn);
            pm.set(ConfusionMatrix.FALSE_POSITIVE, fp);
            pm.set(ConfusionMatrix.FALSE_NEGATIVE, fn);

            ClassificationMetrics cm = new ClassificationMetrics(tn, fp, fn, tp);

            pm.set(PerformanceMeasure.ACCURACY, cm.getAccuracy());
            pm.set(PerformanceMeasure.PRECISION, cm.getPrecision());
            pm.set(PerformanceMeasure.RECALL, cm.getRecall());
            pm.set(PerformanceMeasure.F1SCORE, cm.getF1Score());

            performanceByClass.put(classLabels.get(clsIdx), pm);
        }
        return performanceByClass;
    }


    private void processResult(float[] actual, float[] predicted) {

        if (classLabels.size() == 1) { // if its a binary classifier
            if ((actual[0] == 1) && (predicted[0] >= threshold)) {
                confusionMatrix.inc(1, 1); // tp is at [1, 1]
            } else if ((actual[0] == 0) && (predicted[0] < threshold)) {
                confusionMatrix.inc(0, 0); // tn is at [0, 0]
            } else if ((actual[0] == 0) && (predicted[0] >= threshold)) {
                confusionMatrix.inc(0, 1); // fp is at [0, 1]
            } else if ((actual[0] == 1) && (predicted[0] < threshold)) {
                confusionMatrix.inc(1, 0); // fn is at [1, 0]
            }
        } else { // multi class classifier
            int actualIdx = indexOfMax(actual);
            int predictedIdx = indexOfMax(predicted); // ako su svi nule predictsIdx je od NEGATIVE

            confusionMatrix.inc(actualIdx, predictedIdx);
//            if (predictedIdx != -1 && actualIdx !=-1) {
//                confusionMatrix.inc(actualIdx, predictedIdx);
//            } else if(predictedIdx == -1 && actualIdx !=-1) { // fn
//                for(int i=0; i<classLabels.size(); i++) {
//                    if (i != actualIdx) // fn - da li je ovo dobro?
//                        confusionMatrix.inc(actualIdx, i);
//                    }
//            } else { // tn | predictedIdx == -1 && actualIdx ==-1
//                for(int i=0; i<classLabels.size(); i++) {
//                        confusionMatrix.inc(i, i);
//                }
//            }

        }
    }

    /**
     * Returns index of max element in specified array.
     * If all elements are zero (negative example) returns -1
     *
     * @param array
     * @return index of max value
     */
    private int indexOfMax(float[] array) {
        int maxIdx = 0;
        for (int i = 0; i < array.length; i++) {
         //   if (array[i] >= threshold) {
                if (array[i] > array[maxIdx]) maxIdx = i;
         //   }
        }
        return maxIdx;
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }

    public PerformanceMeasure getTotalAverage() {
        float accuracy = 0, precision = 0, recall = 0, f1score = 0;

        for (PerformanceMeasure pm : performanceByClass.values()) {
            accuracy += pm.get(PerformanceMeasure.ACCURACY);
            recall += pm.get(PerformanceMeasure.RECALL);
            precision += pm.get(PerformanceMeasure.PRECISION);
            f1score += pm.get(PerformanceMeasure.F1SCORE);
        }

        int count = performanceByClass.values().size();

        PerformanceMeasure total = new PerformanceMeasure();
        total.set(PerformanceMeasure.ACCURACY, accuracy / count);
        total.set(PerformanceMeasure.PRECISION, precision / count);
        total.set(PerformanceMeasure.RECALL, recall / count);
        total.set(PerformanceMeasure.F1SCORE, f1score / count);

        return total;
    }

    /**
     * Calculates macro average for the given list of perfromance measures.
     * Used for multi-class classification.
     * Its suitable for balanced classes, but if not micro averaging is more suitable.
     * Micro averaging is summing confusion matrices for individual classes.
     *
     * @param measures
     * @return
     */
    public static PerformanceMeasure averagePerformance(List<PerformanceMeasure> measures) {
        float accuracy = 0, precision = 0, recall = 0, f1score = 0;

        for (PerformanceMeasure pm : measures) {
            accuracy += pm.get(PerformanceMeasure.ACCURACY);
            recall += pm.get(PerformanceMeasure.RECALL);
            precision += pm.get(PerformanceMeasure.PRECISION);
            f1score += pm.get(PerformanceMeasure.F1SCORE);
        }

        int count = measures.size();

        PerformanceMeasure total = new PerformanceMeasure();
        total.set(PerformanceMeasure.ACCURACY, accuracy / count);
        total.set(PerformanceMeasure.PRECISION, precision / count);
        total.set(PerformanceMeasure.RECALL, recall / count);
        total.set(PerformanceMeasure.F1SCORE, f1score / count);

        return total;
    }

    public Map<String, PerformanceMeasure> getPerformanceByClass() {
        return performanceByClass;
    }

    public ConfusionMatrix getConfusionMatrix() {
        return confusionMatrix;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(System.lineSeparator()).append("------------------------------------------------------------------------").append(System.lineSeparator()).
                append("CLASSIFIER EVALUATION RESULTS ").append(System.lineSeparator()).append("------------------------------------------------------------------------").append(System.lineSeparator());
        sb.append("Total classes: ").append(classLabels.size()).append(System.lineSeparator());
//        sb.append("Total correct: ").append(total.correct).append(System.lineSeparator());
//        sb.append("Total incorrect: ").append(total.incorrect).append(System.lineSeparator());
        sb.append("Results by labels").append(System.lineSeparator());

        for (String label : performanceByClass.keySet()) {
            PerformanceMeasure result = performanceByClass.get(label);
//            if (result.get("TotalCorrect") == 0 && result.get("TotalIncorrect") == 0) {
//                continue; // if some of them is negative or nan dont show it
//            }
            sb.append(label).append(": ");
            sb.append(result).append(System.lineSeparator());
        }

        return sb.toString();
    }

}
