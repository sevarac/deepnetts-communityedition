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

/**
 * Confusion matrix container, holds class labels and matrix values.
 * Rows correspond to actual classes, cols to predicted
 *                    Predicted
 *                      F   T
 *  Actual/target  F   TN  FP
 *  Actual/target  T   FN  TP
 *
 */
public class ConfusionMatrix {

    /**
     * Class labels.
     */
    private final String[] classLabels;

    /**
     * Values in confusion matrix.
     */
    private final int[][] values;

    /**
     * Number of classes.
     */
    private final int classCount;

    /**
     * Total number of items classified in this matrix.
     * Sum of all matrix values
     */
    private int totalItems = 0;

    /**
     * Default setting for formating toString
     */
    private static final int STRING_DEFAULT_WIDTH = 7;

    /**
     * Creates a new confusion matrix for specified class labels
     * @param classLabels
     */
    public ConfusionMatrix(String[] classLabels) {

        if (classLabels == null) throw new IllegalArgumentException("Class labels cannot be null!");

//        if (classLabels.length < 2) throw new IllegalArgumentException("Class labels cannot be less then 2!");

        for(String label : classLabels)
            if ((label == null) || label.isEmpty()) throw new IllegalArgumentException("Class label cannot be null or empty String!");

        this.classLabels = classLabels;
        classCount = classLabels.length;
        this.values = new int[classCount][classCount]; // what about negative? must be included in class labels
    }

    /**
     * Returns a value of confusion matrix at specified position.
     * @param predictedIdx predicted class idx - corresponds to row
     * @param actualIdx target/actual class idx  - corresponds to column
     * @return value of confusion matrix at specified position
     */
    public final int get(final int predictedIdx, final int actualIdx) {
       return values[predictedIdx][actualIdx];
    }

    /**
     * Increments matrix value at specified position.
     *
     * @param actualIdx class idx of actual class - corresponds to row
     * @param predictedIdx class idx of predicted class - corresponds to column
     *
     */
    public final void inc(final int actualIdx, final int predictedIdx) {
        values[actualIdx][predictedIdx]++;
        totalItems++;
    }

    public final int getClassCount() {
        return classCount;
    }

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        int maxColumnLenght = STRING_DEFAULT_WIDTH;
        for (String label : classLabels)
            maxColumnLenght = Math.max(maxColumnLenght, label.length());

        // append column names
        builder.append(String.format("%1$" + maxColumnLenght + "s", ""));
        for (String label : classLabels)
            builder.append(String.format("%1$" + maxColumnLenght + "s", label));
        builder.append("\n");

        for (int i = 0; i < values.length; i++) {
            builder.append(String.format("%1$" + maxColumnLenght + "s", classLabels[i]));
            for (int j = 0; j < values[0].length; j++) {
                builder.append(String.format("%1$" + maxColumnLenght + "s", values[i][j]));
            }
            builder.append("\n");

        }
        return builder.toString();
    }

    /**
     * Return true positive for binary classification.
     * How many items that are positive are also classified as positive.
     * @return
     */
    public int getTruePositive() {
        return values[1][1];
    }


    /**
     * Returns true positive for specified class idx for multiclass classification
     * @param clsIdx Index of class for which true positive value is returned
     * @return
     */
    public int getTruePositive(int clsIdx) {
        return values[clsIdx][clsIdx];
    }

    public int getTrueNegative() {
        return values[0][0];
    }
    /*
    https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
*/
    // saberi sva ostala polja, a izuzmi red i kolonu  za zadatu klasu
    // trebalo bi zapravo sabrati sva druga po dijagonali?
    // all non-ci instances that are not classified as c1 - sum everything just skip ci row and col
    public int getTrueNegative(int clsIdx) {
        int trueNegative = 0;

//        for(int i = 0; i < classCount; i++) {
//            if (i == clsIdx) continue;
//            for(int j = 0; j < classCount; j++) {
//                if (j == clsIdx) continue;
//                trueNegative += values[i][j];
//            }
//        }
        trueNegative = values[0][0];
        return trueNegative;
    }

    /**
     * Returns number of false positive classifications.
     * Items that do not belong to specific class, but they are recognized as they do 
     * Only for binary classification
     * @return 
     */
    public int getFalsePositive() {
        return values[0][1];
    }

    // saberi celu clsIdx kolonu samo preskoci clsIdx red
    // all non-clsIdx that are classified/predicted as clsIdx
    public int getFalsePositive(int clsIdx) {
        int falsePositive = 0;

        for(int i=0; i<classCount; i++) {
            if (i == clsIdx) continue; // skip tp value at diagonal
            falsePositive += values[i][clsIdx];
        }

        return falsePositive;
    }

    // Doublechecked: 13.4.19.
    // saberi ceo red  u kome se nalazi zadati clsIdx. znaci clsIdx red a preskoci kolonu clsIdx
    // all clsIdx that are classified as non clasIdx
    public int getFalseNegative(int clsIdx) {
        int falseNegative = 0;

        for(int i=0; i<classCount; i++) {
            if (i == clsIdx) continue; // skip tp value at diagonal
            falseNegative += values[clsIdx][i];
        }

        return falseNegative;
    }

    /**
     * How many positive items has been (falsely) classified as negative.
     * @return How many positive items has been (falsely) classified as negative
     */
    public int getFalseNegative() {
        return values[1][0];
    }

    public final String[] getClassLabels() {
        return classLabels;
    }

    public int getTotalItems() {
        return totalItems;
    }

    public static final String TRUE_POSITIVE="TruePositive";
    public static final String TRUE_NEGATIVE="TrueNegative";
    public static final String FALSE_POSITIVE="FalsePositive";
    public static final String FALSE_NEGATIVE="FalseNegative";

}