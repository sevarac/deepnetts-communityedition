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

import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.List;

/**
 * Container class for all metrics which use confusion matrix for their computation.
 */
public final class ClassificationMetrics {

    float truePositive;
    float trueNegative;
    float falsePositive;
    float falseNegative;
    float total;

    String classLabel; // used when creating classification metrics for specific class for multi class classification problem

   /**
    * Constructs a new measure using arguments
    *
    * @param truePositive
    * @param trueNegative
    * @param falsePositive
    * @param falseNegative
    */
    public ClassificationMetrics(int trueNegative, int falsePositive, int falseNegative, int truePositive) {
        this.truePositive = truePositive;
        this.trueNegative = trueNegative;
        this.falsePositive = falsePositive;
        this.falseNegative = falseNegative;
        this.total = falseNegative + falsePositive + trueNegative + truePositive;
    }

    public ClassificationMetrics(ConfusionMatrix cm) {
        this.truePositive = cm.getTruePositive();
        this.trueNegative = cm.getTrueNegative();
        this.falsePositive = cm.getFalsePositive();
        this.falseNegative = cm.getFalseNegative();
        this.total = falseNegative + falsePositive + trueNegative + truePositive;
    }

    /**
     * Returns class label for
     * @return class labels
     */
    public String getClassLabel() {
        return classLabel;
    }

    public void setClassLabel(String classLabel) {
        this.classLabel = classLabel;
    }

    /**
     * A percent of correct classiifcations/predictions for both positive and negative classes.
     * Answers the question how often is the classifier correct, overall?
     *
     * accuracy = ( tp + tn ) / total
     *
     * @return how often is the classifier correct
     */
    public float getAccuracy() {
        return (truePositive + trueNegative) / total;
    }

    /**
     * A percent of wrong classifications/predictions made.
     * error = (fp + fn) / total
     * error = 1 - accuracy
     *
     * @return classification error rate
     */
    public float getErrorRate() {
        return (falsePositive + falseNegative) / total;
    }


    /**
     * What percent of those predicted as positive are really positive.
     * Answers the question: when it predicts yes, how often is it correct?
     *
     * precision = truePositive / (truePositive + falsePositive)
     *
     * @return when it predicts yes, how often is it correct
     */
    public float getPrecision() {
        return truePositive / (truePositive + falsePositive);
    }

    /**
     * Ratio between those classified as positive compared to those that are actually positive,
     * When ACTUAL class is YES, how often classifier predicts YES?
     * Recall, sensitivity, or true positive rate.
     * @return how often classifier predicts yes, when actual class is yes
     */
    public float getRecall() {
         return truePositive / (truePositive + falsePositive);
    }

    /**
     * Specificity or true negative rate.
     * When it's actually no, how often does it predict no?
     * Ration between those that are classified true negative to those who are actually true negative
     * @return
     */
    public float getSpecificity() {
        return trueNegative / (trueNegative + falseNegative);
    }

   /**
    * Calculates and returns F1 score - harmonic mean of recall and precision
    * f1 = 2  * ( (precision*recall) / (precision+recall))
    *  https://en.wikipedia.org/wiki/F1_score
    * @return f-score
    */
    public float getF1Score() {
        float f1 = 2 * ((getPrecision() * getRecall()) / (getPrecision() + getRecall()));
        return f1;
    }

    /**
     * Returns total number of classifications.
     *
     * @return total number of classifications
     */
    public int getTotal() {
        return (int)total;
    }

    /**
     * When it's actually no, how often does it predict yes?
     * FP/actual no
     * @return
     */
    public float getFalsePositiveRate() {
        return falsePositive / (falsePositive + trueNegative);
    }

    /**
     * When its actually yes, how often does it predicts no
     * @return
     */
    public float getFalseNegativeRate() {
        return falseNegative / (falseNegative + truePositive );
    }

    /**
     * When its actauly no, how often it is classified as yes
     * @return
     */
    public float getFalseDiscoveryRate() {
        return falsePositive / (falsePositive+truePositive);
    }

//        Prevalence: How often does the yes condition actually occur in our sample?
//        actual yes/total = 105/165 = 0.64
    public float positiveFreqency() {
        return (truePositive+falseNegative) / total;
    }

    public float negativeFreqency() {
        return (trueNegative+falsePositive) / total;
    }

    // dovde sam prekontrolisao sve formule!
    //-------------------------------------------------------------------------------

    /**
     * Returns the F-score. When recall and precision are zero, this method will
     * return 0.
     *  https://en.wikipedia.org/wiki/F1_score
     * @param beta
     * @return f-score
     */
    public float getFScore(int beta) {
        float f = ((beta * beta + 1) * getPrecision() * getRecall())
                / (float)(beta * beta * getPrecision() + getRecall());
        if (Double.isNaN(f))
            return 0;
        else
            return f;
    }


    // http://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    // The F1 metric is not a suitable method of combining precision and recall i
    //  measure of the quality of binary (two-class) classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
    public double getMatthewsCorrelationCoefficient() {
        return (truePositive * trueNegative - falsePositive * falseNegative) /
                (sqrt((truePositive + falsePositive) * (truePositive + falseNegative) * (trueNegative + falsePositive) * (trueNegative + falseNegative)));
    }


    public double getBalancedClassificationRate() {
        if (trueNegative == 0 && falsePositive == 0)
            return truePositive / (truePositive + falseNegative);
        if (truePositive == 0 && falseNegative == 0)
            return trueNegative / (trueNegative + falsePositive);

        return 0.5 * (truePositive / (truePositive + falseNegative) + trueNegative / (trueNegative + falsePositive));
    }



    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append("Class: "+classLabel).append("\n");
        sb.append("Total items: ").append(getTotal()).append("\n");
        sb.append("True positive:").append(truePositive).append("\n");
        sb.append("True negative:").append(trueNegative).append("\n");
        sb.append("False positive:").append(falsePositive).append("\n");
        sb.append("False negative:").append(falseNegative).append("\n");
        sb.append("Accuracy (ACC): ").append(getAccuracy()).append("\n");
        sb.append("Sensitivity (TPR): ").append(getRecall()).append("\n");
        sb.append("Specificity (TNR): ").append(getSpecificity()).append("\n");
        sb.append("Fall-out (FPR): ").append(getFalsePositiveRate()).append("\n");
        sb.append("False negative rate (FNR): ").append(getFalseNegativeRate()).append("\n");
        sb.append("Precision (PPV): ").append(getPrecision()).append("\n");
        sb.append("Recall: ").append(getRecall()).append("\n");
        sb.append("F1 Score: ").append(getF1Score()).append("\n");
        sb.append("False discovery rate (FDR): ").append(getFalseDiscoveryRate()).append("\n");
        sb.append("Matthews correlation Coefficient (MCC): ").append(getMatthewsCorrelationCoefficient()).append("\n");
        return sb.toString();
    }


    public static class Stats {
        public double accuracy=0;
        public double precision=0;
        public double recall=0;
        public double fScore=0;
        public double mserror=0;
        public double correlationCoefficient = 0;

        @Override
        public String toString() {
            return "Stats{" + "accuracy=" + accuracy + ", precision=" + precision + ", recall=" + recall + ", fScore=" + fScore + ", mserror=" + mserror + ", corelationCoefficient=" + correlationCoefficient + '}';
        }


    }

    /**
     * Creates classification metrics from the given confusion matrix.
     * Creates an array of ClassificationMetrics objects one for each class.
     *
     * @param confusionMatrix
     * @return classification metrics
     */
    public static ClassificationMetrics[] createFrom(ConfusionMatrix confusionMatrix) {

        int classCount = confusionMatrix.getClassCount();
        if (classCount == 2) { // binary classification
            ClassificationMetrics[] measures = new ClassificationMetrics[1];
            String[] classLabels = confusionMatrix.getClassLabels();

                int tp = confusionMatrix.getTruePositive();
                int tn = confusionMatrix.getTrueNegative();
                int fp = confusionMatrix.getFalsePositive();
                int fn = confusionMatrix.getFalseNegative();

            measures[0] = new ClassificationMetrics(tn, fp, fn, tp);
            measures[0].setClassLabel(classLabels[0]);

            return measures;

        } else { // multiclass classification
            ClassificationMetrics[] measures = new ClassificationMetrics[classCount];
            String[] classLabels = confusionMatrix.getClassLabels();

            for(int clsIdx=0; clsIdx<confusionMatrix.getClassCount(); clsIdx++) { // for each class
                // ove metode mozda ubaciti u matricu Confusion matrix - najbolje tako
                int tp = confusionMatrix.getTruePositive(clsIdx);
                int tn = confusionMatrix.getTrueNegative(clsIdx);
                int fp = confusionMatrix.getFalsePositive(clsIdx);
                int fn = confusionMatrix.getFalseNegative(clsIdx);

                measures[clsIdx] = new ClassificationMetrics(tn, fp, fn, tp);
                measures[clsIdx].setClassLabel(classLabels[clsIdx]);
            }
            return measures;
        }

    }



    /**
     *
     * @param results list of different metric results computed on different sets of data
     * @return average metrics computed different MetricResults
     */
    public static ClassificationMetrics.Stats average(ClassificationMetrics[] results) {
        List<String> classLabels = new ArrayList<>();
         ClassificationMetrics.Stats average = new ClassificationMetrics.Stats();
          double count = 0;
            for (ClassificationMetrics cm : results) {
                average.accuracy += cm.getAccuracy();
                average.precision += cm.getPrecision();
                average.recall += cm.getRecall();
                average.fScore += cm.getF1Score();
//                average.mserror += er.getMeanSquareError();

                if(!classLabels.contains(cm.getClassLabel()))
                    classLabels.add(cm.getClassLabel());
            }
            count++;

        count = count * classLabels.size(); // * classes count
        average.accuracy = average.accuracy / count;
        average.precision = average.precision / count;
        average.recall = average.recall / count;
        average.fScore = average.fScore / count;
        average.mserror = average.mserror / count;

        return average;
    }

    /**
     *
     * @param results list of different metric results computed on different sets of data
     * @return maximum metrics computed different MetricResults
     */
//    public static ClassificationMetrics maxFromMultipleRuns(List<ClassificationMetrics> results) {
//        double maxAccuracy = 0;
//        double maxError = 0;
//        double maxPrecision = 0;
//        double maxRecall = 0;
//        double maxFScore = 0;
//
//        for (ClassificationMetrics metricResult : results) {
//            maxAccuracy = Math.max(maxAccuracy, metricResult.getAccuracy());
//            maxError = Math.max(maxError, metricResult.getError());
//            maxPrecision = Math.max(maxPrecision, metricResult.getPrecision());
//            maxRecall = Math.max(maxRecall, metricResult.getRecall());
//            maxFScore = Math.max(maxFScore, metricResult.getFScore());
//        }
//
//        ClassificationMetrics averageMetricsResult = new ClassificationMetrics();
//
//        averageMetricsResult.accuracy = maxAccuracy;
//        averageMetricsResult.error = maxError;
//        averageMetricsResult.precision = maxPrecision;
//        averageMetricsResult.recall = maxRecall;
//        averageMetricsResult.fScore = maxFScore;
//
//        return averageMetricsResult;
//    }



//    private static double[] createFScoresForEachClass(double[] precisions, double[] recalls) {
//        double[] fScores = new double[precisions.length];
//
//        for (int i = 0; i < precisions.length; i++) {
//            fScores[i] = 2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i]);
//        }
//
//        return fScores;
//    }


//    private static double safelyDivide(double x, double y) {
//        double divisor = x == 0.0 ? 1 : x;
//        double divider = y == 0.0 ? 1.0 : y;
//        return divisor / divider;
//    }


}