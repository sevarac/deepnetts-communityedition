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

package deepnetts.net.train;

import deepnetts.data.BasicDataSet;
import deepnetts.data.DataSet;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.Evaluator;
import deepnetts.eval.EvaluationMetrics;
import deepnetts.net.NeuralNetwork;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.lang3.SerializationUtils;

/**
 * Split data set into k parts of equal sizes (folds)
 * Train with data from k-1 folds(parts), and test with 1 fold, repeat k times each with different test fold.
 *
 * @author Zoran
 */
public class KFoldCrossValidation {

    private int splitsNum; 
    private NeuralNetwork neuralNetwork; 
    private BackpropagationTrainer trainer; 
    private DataSet<?> dataSet; 
    private Evaluator<NeuralNetwork, DataSet<?>> evaluator; 
    private final List<NeuralNetwork> trainedNetworks = new ArrayList<>();


    public EvaluationMetrics runCrossValidation() {
        List<EvaluationMetrics> measures = new ArrayList<>();
        DataSet[] folds = dataSet.split(splitsNum);

        for (int testFoldIdx = 0; testFoldIdx < splitsNum; testFoldIdx++) {
            DataSet testSet = folds[testFoldIdx];
            DataSet trainingSet = new BasicDataSet(((BasicDataSet)dataSet).getInputsNum(), ((BasicDataSet)dataSet).getOutputsNum());
            trainingSet.setColumnNames( ((BasicDataSet)dataSet).getColumnNames());
            for (int trainFoldIdx = 0; trainFoldIdx < splitsNum; trainFoldIdx++) {
                if (trainFoldIdx == testFoldIdx) continue;
                trainingSet.addAll(folds[trainFoldIdx]);
            }

            NeuralNetwork neuralNet = SerializationUtils.clone(this.neuralNetwork);

            trainer.train(trainingSet); 
            EvaluationMetrics pe = evaluator.evaluate(neuralNet, testSet); 
            measures.add(pe);
            trainedNetworks.add(neuralNet);
        }

        return ClassifierEvaluator.averagePerformance(measures);
    }

    public List<NeuralNetwork> getTrainedNetworks() {
        return trainedNetworks;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        KFoldCrossValidation kFoldCV = new KFoldCrossValidation();

        public Builder splitsNum(int k) {
           kFoldCV.splitsNum = k;
           return this;
        }

        public Builder model(NeuralNetwork neuralNet) {
            kFoldCV.neuralNetwork = neuralNet;
            return this;
        }

        public Builder trainer(BackpropagationTrainer trainer) {
            kFoldCV.trainer = trainer;
            return this;
        }

        public Builder dataSet(DataSet dataSet) {
            kFoldCV.dataSet = dataSet;
            return this;
        }

        public Builder evaluator(Evaluator<NeuralNetwork, DataSet<?>> evaluator) {
            kFoldCV.evaluator = evaluator;
            return this;
        }

        public KFoldCrossValidation build() {
            return kFoldCV;
        }

    }
}
