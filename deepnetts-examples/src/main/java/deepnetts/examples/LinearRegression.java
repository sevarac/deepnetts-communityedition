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
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import java.io.File;
import java.io.IOException;

/**
 * Minimal example for simple linear regression using FeedForwardNetwork.
 * Fits a straight line through the data.
 * Uses a single addLayer with one output and linear activation function, and Mean Squared Error for Loss function.
 * You can use linear regression to roughly estimate a global trend in data.
 *
 * TODO: dont print accuracy for regression problems!
 *
 * predicting the total payment for all claims in thousands of Swedish Kronor, given the total number of claims.
 *
 * @author Zoran Sevarac
 */
public class LinearRegression {

    public static void main(String[] args) throws IOException {

        String datasetFile = "datasets/SweedenAutoInsurance.csv";
        int inputsCount = 1;
        int outputsCount = 1;
        String delimiter = ",";

        DataSet dataSet = BasicDataSet.fromCSVFile(new File(datasetFile), inputsCount, outputsCount, delimiter);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(1)
                .addOutputLayer(1, ActivationType.LINEAR)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

        BackpropagationTrainer trainer = new BackpropagationTrainer();
        trainer.setMaxError(0.01f)
               .setMaxEpochs(100)
               .setLearningRate(0.1f)
               .train(neuralNet, dataSet);

    }

}
