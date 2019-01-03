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
import deepnetts.util.RandomGenerator;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Iris Classification Problem. Using sigmoid activation in output addLayer by
 * default and mse as a loss function.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class IrisClassificationMoreUserFriendlyAPI {

    private static final Logger LOGGER = LogManager.getLogger(IrisClassificationMoreUserFriendlyAPI.class.getName());
    
    public static void main(String[] args) throws DeepNettsException, IOException {
        int inputsNum = 4;
        int outputsNum = 3;
       
        RandomGenerator.getDefault().initSeed(123);        
        
        // load iris data set from csv file
    //    DataSet dataSet = BasicDataSet.fromCSVFile("datasets/iris_data_normalised.txt", inputsNum, outputsNum); // perfrom OneHotEncoding, and normalization, optionally missing values and balansing
        DataSet dataSet = deepnetts.data.DataSets.readCsv("datasets/iris_data_normalised.txt", inputsNum, outputsNum); // perfrom OneHotEncoding, and normalization, optionally missing values and balansing
                
//        dataSet.shuffle();
        DataSet[] dataSets = dataSet.split(0.6);    // shuffle before split by default init main random seed before everythng else. this split could also go to datasets utility class

        // create multi layer perceptron with specified settings
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(inputsNum)   //  rename to inputs?
                .addDenseLayer(8) // , ActivationType.SIGMOID by defaylt? or better yse Tanh or relu?
                .addOutputLayer(outputsNum, ActivationType.SIGMOID) // rename to outputs?
                .lossFunction(LossType.MEAN_SQUARED_ERROR)
                .build();

//        neuralNet.train(dataSets[0]);    // getTrainer().train() 
        
        // neuralNet.getTrainer(). ...
        // neuralNet.setTrainer(trainer);
        
        //Evaluators.evaluateClassifier(neuralNet, dataSets[1]);
        //Evaluators.evaluateRegressor(neuralNet, dataSets[1]);
        
        //neuralNet.getTrainer().train(neuralnet, dataset)      implements Trainable, TrainerProvider<T>
         
        // neuralNet.train(trainingSet, validationSet);
        // create a trainer object with specified settings
//        BackpropagationTrainer trainer = new BackpropagationTrainer();
//        trainer.setMaxError(0.06f)                  // should be default
//                .setLearningRate(0.1f)             // should be default
//                .setOptimizer(OptimizerType.SGD) // should be default
//                .setBatchMode(false);   // should be default


        // train the network
//        trainer.train(neuralNet, dataSets[0]);
        
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
