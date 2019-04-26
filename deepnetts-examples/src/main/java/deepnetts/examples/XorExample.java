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
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */
package deepnetts.examples;

import deepnetts.data.BasicDataSetItem;
import deepnetts.data.BasicDataSet;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;

/**
 * Solve XOR problem to confirm that backpropagation is working, and that it can
 * solve the simplest nonlinear problem.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class XorExample {

    public static void main(String[] args) throws DeepNettsException {

        DataSet dataSet = ExampleDataSets.xor();
        dataSet.setColumnNames(new String[] {"col1", "col2", "col3"});

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(2)
                .addFullyConnectedLayer(3, ActivationType.TANH)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
//                .randomSeed(123)
                .build();
        
//        neuralNet.getTrainer().setLearningRate(0.9f);
        neuralNet.setOutputLabels("output");
        
//        neuralNet.train(dataSet);

        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
        trainer.setMaxError(0.01f);
        trainer.setLearningRate(0.9f);
        trainer.train(dataSet);
    }



}
