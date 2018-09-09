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
 * TODO: if class count == 2 use binary classifier else, its multi class
 * classifier!
 *
 * http://www.ritchieng.com/machine-learning-evaluate-classification-model/
 * http://scikit-learn.org/stable/modules/model_evaluation.html
 * http://notesbyanerd.com/2014/12/17/multi-class-performance-measures/
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

    private float threshold = 0.5f; // this should go into classifier

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
        classLabels = Arrays.asList(testSet.getOutputLabels()); // get output labels
        // if class labels are empty create class1, class2, classk ....
        init();

        //  wrap neural network with classifier interface, setInput from param, and return output
        // I need a method that wraps modelinto a classifier Classifier.fromNeuralNetwork(neuralNet)
        for (DataSetItem item : testSet) {
            neuralNet.setInput(item.getInput());
            neuralNet.forward();
            final float[] output = neuralNet.getOutput();
            processResult(output, item.getTargetOutput());
        }

        if (classLabels.size() == 1) {  // for binary classification
            return createBinaryPerformanceMeasures();
        } else {    // for multi class classification
            createMultiClassPerformanceMeasures();
            return getTotalAverage(); // najbolje da ovde vracam total a sa posebnom metodom da uzimam bu class
        }
    }

    private PerformanceMeasure createBinaryPerformanceMeasures() {
        PerformanceMeasure pm = new PerformanceMeasure();

        int tp = confusionMatrix.getTruePositive();
        int tn = confusionMatrix.getTrueNegative();
        int fp = confusionMatrix.getFalsePositive();
        int fn = confusionMatrix.getFalseNegative();

        ClassificationMetrics cm = new ClassificationMetrics(tp, tn, fp, fn);

        pm.set("TotalClasses", classLabels.size());
        pm.set("TotalItems", cm.getTotal());

        pm.set("TruePositive", tp);
        pm.set("TrueNegative", tn);
        pm.set("FalsePositive", fp);
        pm.set("FalseNegative", fn);

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

            ClassificationMetrics cm = new ClassificationMetrics(tp, tn, fp, fn);

            pm.set("TruePositive", tp);
            pm.set("TrueNegative", tn);
            pm.set("FalsePositive", fp);
            pm.set("FalseNegative", fn);

            pm.set(PerformanceMeasure.ACCURACY, cm.getAccuracy());
            pm.set(PerformanceMeasure.PRECISION, cm.getPrecision());
            pm.set(PerformanceMeasure.RECALL, cm.getRecall());
            pm.set(PerformanceMeasure.F1SCORE, cm.getF1Score());

            performanceByClass.put(classLabels.get(clsIdx), pm);
        }
        return performanceByClass;
    }

    // https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
    // http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    private void processResult(float[] predictedOutput, float[] targetOutput) {

        if (classLabels.size() == 1) { // if its a binary classifier
            if ((predictedOutput[0] >= threshold) && (targetOutput[0] == 1)) {
                confusionMatrix.inc(0, 0); // tp is at [0, 0]
            } else if ((predictedOutput[0] < threshold) && (targetOutput[0] == 0)) {
                confusionMatrix.inc(1, 1); // tn is at [0, 0]
            } else if ((predictedOutput[0] >= threshold) && (targetOutput[0] == 0)) {
                confusionMatrix.inc(0, 1); // fp is at [0, 1]
            } else if ((predictedOutput[0] < threshold) && (targetOutput[0] == 1)) {
                confusionMatrix.inc(1, 0); // fn is at [1, 0]
            }
        } else { // multi class classifier
            int actualIdx = indexOfMax(targetOutput);
//            String actualClass = null;
//
//            if (!isNegativeTarget(targetOutput)) {
//                actualClass = classLabels.get(actualIdx);
//            } else {
//                actualClass = NEGATIVE; // ako su svi nue, ond aje negativan primer
//            }

            int predictedIdx = indexOfMax(predictedOutput); // ako su svi nule predictsIdx je od NEGATIVE
//            String predictedClass = null;
//            if (predictedOutput[predictedIdx] >= threshold) {
//                predictedClass = classLabels.get(predictedIdx);
//            } else {
//                predictedClass = NEGATIVE;
//            }

            confusionMatrix.inc(predictedIdx, actualIdx);
        }
    }

    /**
     * Returns index of max element in specified array.
     *
     * @param array
     * @return index of max value
     */
    private int indexOfMax(float[] array) {
        int maxIdx = 0;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > array[maxIdx]) {
                maxIdx = i;
            }
        }
        return maxIdx;
    }

    // target is negative if all target outputs are 0
    private boolean isNegativeTarget(float[] array) {
        for (int i = 0; i < array.length; i++) {
            if (array[i] != 0) {
                return false;
            }
        }

        return true;
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
            if (result.get("TotalCorrect") == 0 && result.get("TotalIncorrect") == 0) {
                continue; // if some of them is negative or nan dont show it
            }
            sb.append(label).append(": ");
            sb.append(result).append(System.lineSeparator());
        }

        return sb.toString();
    }

}
