/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */

package deepnetts.examples;

import deepnetts.data.BasicDataSet;
import deepnetts.data.DataSet;
//import deepnetts.eval.Evaluators;
import deepnetts.eval.PerformanceMeasure;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import java.io.IOException;

/**
 * Minimal example for logistic regression using FeedForwardNetwork.
 * Can perform binary classification (single output true/false)
 Uses only input and output addLayer with sigmoid activation function
 Just specify number of inputs and provide data set
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class LogisticRegression {


    public static void main(String[] args) throws IOException {

        BasicDataSet dataSet =(BasicDataSet) BasicDataSet.fromCSVFile("datasets/sonar.csv", 60, 1, ","); // get data from some file or method

        DataSet[] trainAndTestSets = dataSet.split(0.6, 0.4);

        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(60)
               // .addFullyConnectedLayer(3, ActivationType.TANH)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
                               trainer.setLearningRate(0.01f)
                                       .setMaxError(0.1f)
                                      .train(trainAndTestSets[0]);

        PerformanceMeasure pm = Evaluators.evaluateClassifier(neuralNet, trainAndTestSets[1]);
        System.out.println(pm);

    }

}
