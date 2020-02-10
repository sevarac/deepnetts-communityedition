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
import javax.visrec.ml.eval.Evaluator;
import java.util.List;
import deepnetts.data.MLDataItem;

/**
 * Evaluates regressor neural network for specified data set.
 * Assumes only one output at the moment.
 * 
 * @author Zoran Sevarac
 */
public class RegresionEvaluator implements Evaluator<NeuralNetwork, DataSet<MLDataItem>> {

    @Override
    public EvaluationMetrics evaluate(NeuralNetwork neuralNet, DataSet<MLDataItem> testSet) {
        EvaluationMetrics pe = new EvaluationMetrics();

        MeanSquaredError mse = new MeanSquaredError();
        float tss=0;
        int numInputs = testSet.get(0).getInput().size();
        int numItems= testSet.size();
        float targetMean = mean(testSet);

        for (MLDataItem item : testSet) {
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

    private float mean(DataSet<? extends MLDataItem> testSet) {
        float mean=0;
        for(MLDataItem ditem : testSet) {
            mean += ditem.getTargetOutput().get(0);
        }
        return mean / testSet.size();
    }
    
    public static EvaluationMetrics averagePerformance(List<EvaluationMetrics> measures) {
        float mse = 0, rse = 0, r2 = 0, fstat = 0;

        for (EvaluationMetrics em : measures) {
            mse += em.get(EvaluationMetrics.MEAN_SQUARED_ERROR);
            r2 += em.get(EvaluationMetrics.RESIDUAL_STANDARD_ERROR);
            rse += em.get(EvaluationMetrics.R_SQUARED);
            fstat += em.get(EvaluationMetrics.F_STAT);
        }

        int count = measures.size();

        EvaluationMetrics total = new EvaluationMetrics();
        total.set(EvaluationMetrics.MEAN_SQUARED_ERROR, mse / count);
        total.set(EvaluationMetrics.RESIDUAL_STANDARD_ERROR, rse / count);
        total.set(EvaluationMetrics.R_SQUARED, r2 / count);
        total.set(EvaluationMetrics.F_STAT, fstat / count);

        return total;
    }    

}
