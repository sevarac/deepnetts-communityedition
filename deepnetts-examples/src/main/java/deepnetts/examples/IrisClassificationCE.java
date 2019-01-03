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
import deepnetts.data.BasicDataSet;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.DeepNettsException;
import java.io.File;
import java.io.IOException;

/**
 * Iris Classification Problem. This example is using Softmax activation in
 * output layer and Cross Entropy Loss function. Overfits the iris data set
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class IrisClassificationCE {

    public static void main(String[] args) throws DeepNettsException, IOException {

        // load iris data  set
        DataSet dataSet = BasicDataSet.fromCSVFile(new File("datasets/iris_data_normalised.txt"), 4, 3, ",");
        dataSet.shuffle();

        // create instance of multi addLayer percetpron using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(4)
                .addDenseLayer(9, ActivationType.TANH) // 20 hid 28 epochs, 10 51, 30 hid, 35 epochs, 15 hid 46 epochs, 5 hid 62 epochs, 3 hid 41 epochs
                .addOutputLayer(3, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        // create and configure instanceof backpropagation trainer
        BackpropagationTrainer trainer = new BackpropagationTrainer();
        trainer.setMaxError(0.03f);
        trainer.setLearningRate(0.1f);
        trainer.setBatchMode(false);
        trainer.setMomentum(0.3f);
        trainer.setOptimizer(OptimizerType.MOMENTUM);
        trainer.setMaxEpochs(10000);
        trainer.train(neuralNet, dataSet);
    }

}
