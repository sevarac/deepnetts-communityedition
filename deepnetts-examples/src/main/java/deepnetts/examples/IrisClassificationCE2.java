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

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSet;
import deepnetts.data.DataSets;
import deepnetts.data.util.MaxNormalizer;
import deepnetts.eval.Evaluators;
import deepnetts.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Iris Classification Problem. This example is using Softmax activation in
 * output addLayer and Cross Entropy Loss function.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class IrisClassificationCE2 {

    private static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    public static void main(String[] args) throws DeepNettsException, IOException {
        // load iris data  set
        DataSet dataSet = DataSets.readCsv("datasets/iris.csv", 4, 3, true, ",");
        DataSet[] dataSets = dataSet.split(0.7, 0.3); // provide random generator in order to do spliting the same way

        MaxNormalizer norm = new MaxNormalizer(dataSet);
        norm.normalize(dataSets[0]);   
        norm.normalize(dataSets[1]);
        
        // create instance of multi addLayer percetpron using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(4)
                .addFullyConnectedLayer(12, ActivationType.TANH)
          //      .addFullyConnectedLayer(8, ActivationType.TANH)
            //    .addFullyConnectedLayer(6, ActivationType.TANH)
                .addOutputLayer(3, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123).
                build();

//        neuralNet.train(dataSets[0]);

        // create and configure instanceof backpropagation trainer
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
        trainer.setMaxError(0.02f);
        trainer.setLearningRate(0.02f);
        trainer.setBatchMode(false);
        trainer.setMomentum(0.9f);  // sa celim data setom radi i sa momentumom
        trainer.setOptimizer(OptimizerType.MOMENTUM);
        trainer.setMaxEpochs(10000);
        trainer.train(dataSets[0]);


        EvaluationMetrics pm = Evaluators.evaluateClassifier(neuralNet, dataSets[1]);
        
        LOGGER.info("Classification performance measure" + System.lineSeparator());
        LOGGER.info(pm);
    }

}
