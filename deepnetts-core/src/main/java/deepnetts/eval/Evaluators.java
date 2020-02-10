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

import deepnetts.net.NeuralNetwork;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.data.MLDataItem;

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
    public static EvaluationMetrics evaluateRegressor(NeuralNetwork<?> neuralNet, DataSet<MLDataItem> testSet) {
        RegresionEvaluator eval = new RegresionEvaluator();
        return eval.evaluate(neuralNet, testSet);
    }

    /**
     *
     * @param neuralNet
     * @param testSet
     * @return classification performance measure
     */
    public static EvaluationMetrics evaluateClassifier(NeuralNetwork<?> neuralNet, DataSet<MLDataItem> testSet) {
        ClassifierEvaluator eval = new ClassifierEvaluator();
        return eval.evaluate(neuralNet, testSet);
    }

}