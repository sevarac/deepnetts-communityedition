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
package deepnetts.examples;

import deepnetts.data.DataSet;
import deepnetts.data.DataSets;
import deepnetts.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;

/**
 * Spam  Classification Problem. This example is using Softmax activation in
 * output layer and Cross Entropy Loss function. Overfits the iris data set
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class SpamClassifier {

    public static void main(String[] args) throws DeepNettsException, IOException {

        // load spam data  set from csv file
        DataSet dataSet = DataSets.readCsv("datasets//spam.csv", 57, 1, true);             

        // create instance of feed forward neural network its builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(57)
                .addFullyConnectedLayer(25, ActivationType.TANH)
                .addOutputLayer(1, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(456)
                .build();

        // start training
        neuralNet.train(dataSet);
        
        // test network /  evaluate classification accuracy
        EvaluationMetrics em = neuralNet.test(dataSet);
        System.out.println(em);
    }
}