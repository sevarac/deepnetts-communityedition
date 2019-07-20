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

    private int splitsNum; //number of folds, typically  5 or 10 used, pg. 184    // numSplits
    private NeuralNetwork neuralNetwork; // arhitektura neuronske mreze
    private BackpropagationTrainer trainer; // algoritam za trening sa svim svojim podesavanjima podesenim
    private DataSet<?> dataSet; // data set koji se deli
    private Evaluator<NeuralNetwork, DataSet<?>> evaluator; // mogao bi u logu da ispisuje rezultate evaluacije kao json
    private final List<NeuralNetwork> trainedNetworks = new ArrayList<>();

    // posto treba da radi multi threaded training, najbolje da ovo radi u istom thread-u, da ne bi rasipao threadove
    // najbolje da kreira trening i test set od foldova

    // podeli data set na k jednakih foldova
    // sa k-1 treniraj sa onim preostalim testiraj i izracunaj prosecne mere  performansi (MSE i klasifikacija)

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

            // clone the original network each time before training - create a new instace that will be added to trainedNetworks
            NeuralNetwork neuralNet = SerializationUtils.clone(this.neuralNetwork);

            trainer.train(trainingSet); // napravi da trainer moze da sa istim parametrima pozove novu mrezu!!!!! ovo je problem, trainer zahteva novu instancu neuralNet ovde!!!
            EvaluationMetrics pe = evaluator.evaluate(neuralNet, testSet); // Peturn an instance of PerformanceMeaseure here
            measures.add(pe);
            trainedNetworks.add(neuralNet);
        }
        // get final evaluation results - avg performnce of all test sets - use some static method to get that
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
