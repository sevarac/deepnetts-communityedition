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

import deepnetts.eval.ClassifierEvaluator;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import java.util.Map;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Iris Classification Problem. Using sigmoid activation in output addLayer by
 * default and mse as a loss function.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class IrisClassificationMseWithRegularization {

    private static final Logger LOGGER = LogManager.getLogger(IrisClassificationMseWithRegularization.class.getName());
    
    public static void main(String[] args) throws DeepNettsException, IOException {
        int inputsNum = 4;
        int outputsNum = 3;
                
        // load iris data set from csv file
        //DataSet dataSet = BasicDataSet.fromCSVFile("datasets/iris_data_normalised.txt", 4, 3);
        DataSet dataSet = deepnetts.data.DataSets.readCsv("datasets/iris_data_normalised.txt", inputsNum, outputsNum,true); // perfrom OneHotEncoding, and normalization, optionally missing values and balansing
        dataSet.shuffle();
        DataSet[] dataSets = dataSet.split(0.6);        

        // create multi layer perceptron with specified settings
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(4)
                .addFullyConnectedLayer(8)
                .addOutputLayer(3, ActivationType.SIGMOID)
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .randomSeed(123)
                .build();

        //neuralNet.getTrainer().train(neuralnet, dataset)      implements Trainable, TrainerProvider<T>
        // neuralNet.train(trainingSet);    // getTrainer().train() 
        // neuralNet.train(trainingSet, validationSet);
        // create a trainer object with specified settings
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
        trainer.setMaxError(0.03f)                  // should be default
                .setMaxEpochs(21000)
                .setLearningRate(0.01f)             // should be default
                .setL2Regularization(0.001f)
                .setOptimizer(OptimizerType.SGD) // should be default
                .setBatchMode(false);   // should be default    provide logging options - what to log - logMiniBatch

        // train the network
        trainer.train(dataSets[0]);
        
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        EvaluationMetrics pm = evaluator.evaluate(neuralNet, dataSets[1]);
        LOGGER.info("------------------------------------------------");
        LOGGER.info("Classification performance measure" + System.lineSeparator());
        LOGGER.info(pm);
        Map<String, EvaluationMetrics> byClass = evaluator.getPerformanceByClass();
        byClass.entrySet().stream().forEach((entry) -> {
            LOGGER.info("Class " + entry.getKey() + ":");
            LOGGER.info(entry.getValue());
            LOGGER.info("----------------");
        });        
    }

}