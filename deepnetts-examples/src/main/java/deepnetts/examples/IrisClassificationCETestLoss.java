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
import deepnetts.data.BasicDataSet;
import deepnetts.data.DataSet;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.PerformanceMeasure;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.DeepNettsException;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Iris Classification Problem. This example is using Softmax activation in
 * output addLayer and Cross Entropy Loss function.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class IrisClassificationCETestLoss {

    private static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    public static void main(String[] args) throws DeepNettsException, IOException {
        // load iris data  set
        DataSet dataSet = BasicDataSet.fromCsv("datasets/iris_data_normalised.txt", 4, 3);
        dataSet.shuffle(); // do the shuffling inside the split method automaticaly! how to specify random seed for shuffling?
        DataSet[] dataSets = dataSet.split(65, 35);
        // dataSet.normalize();// Norm.MAX Norm.RANGE Norm.ZSCORE, i overload gde kao parametar prihvata normalizator?

        // create instance of multi addLayer percetpron using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(4)
                .addDenseLayer(16, ActivationType.TANH)
                .addOutputLayer(3, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123).
                build();

        // create and configure instanceof backpropagation trainer
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
        trainer.setMaxError(0.01f);
        trainer.setLearningRate(0.03f);
        trainer.setBatchMode(false);
        trainer.setMomentum(0.9f);
        trainer.setOptimizer(OptimizerType.MOMENTUM);
        trainer.setMaxEpochs(10000);
        trainer.train(dataSets[0], dataSets[1]);

        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        PerformanceMeasure pm = evaluator.evaluatePerformance(neuralNet, dataSets[1]);
        LOGGER.info("------------------------------------------------");
        LOGGER.info("Classification performance measure" + System.lineSeparator());
        LOGGER.info(pm);
        Map<String, PerformanceMeasure> byClass = evaluator.getPerformanceByClass();
        byClass.entrySet().stream().forEach((entry) -> {
            LOGGER.info("Class " + entry.getKey() + ":");
            LOGGER.info(entry.getValue());
            LOGGER.info("----------------");
        });

    }

}
