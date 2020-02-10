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
package examplesbak;

import deepnetts.core.DeepNetts;
import deepnetts.data.DataSets;
import deepnetts.eval.Evaluators;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Iris Classification Problem. This example is using Softmax activation in
 * output addLayer and Cross Entropy Loss function.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class IrisClassificationWithDropout {

    private static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    public static void main(String[] args) throws DeepNettsException, IOException {
        // load iris data  set
        DataSet dataSet = DataSets.readCsv("datasets/iris_data_normalised.txt", 4, 3, true, ",");
        DataSet[] trainTestSet = dataSet.split(0.6, 0.4);

        // create instance of multi addLayer percetpron using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(4)
                .addFullyConnectedLayer(9, ActivationType.TANH)
                .addOutputLayer(3, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123).
                build();

        // create and configure instanceof backpropagation trainer
        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setMaxError(0.03f)
                .setLearningRate(0.01f)
                .setBatchMode(false)
                .setMomentum(0.9f)
                .setOptimizer(OptimizerType.MOMENTUM)
                .setMaxEpochs(10000)
                .setDropout(0.3f);
        
        trainer.train(trainTestSet[0]);


        EvaluationMetrics em = Evaluators.evaluateClassifier(neuralNet, trainTestSet[1]);

        LOGGER.info("------------------------------------------------");
        LOGGER.info("Classifier evaluation metrics" + System.lineSeparator());
        LOGGER.info(em);

    }

}
