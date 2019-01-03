/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 */
package deepnetts.examples;

import deepnetts.data.DataSet;
import deepnetts.data.BasicDataSet;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;

/**
 * Iris Classification Problem. This example is using Softmax activation in
 * output layer and Cross Entropy Loss function. Overfits the iris data set
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class QuickStart {

    public static void main(String[] args) throws DeepNettsException, IOException {
        // load data  set from csv file
        DataSet dataSet = BasicDataSet.fromCSVFile(new File("datasets/iris_data_normalised.txt"), 4, 3, ",");
        dataSet.shuffle();

        // create instance of multi addLayer percetpron using builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(4)
                .addDenseLayer(10, ActivationType.TANH)
                .addOutputLayer(3, ActivationType.SOFTMAX)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        // create and configure instanceof backpropagation trainer
        BackpropagationTrainer trainer = new BackpropagationTrainer();
        trainer.setMaxError(0.05f);
        trainer.setMaxEpochs(10000);
        trainer.setLearningRate(0.01f);

        
        // run training
        trainer.train(neuralNet, dataSet);
        
        // save trained network to file
        FileIO.writeToFile(neuralNet, "myNeuralNet.dnet");
    }
}
