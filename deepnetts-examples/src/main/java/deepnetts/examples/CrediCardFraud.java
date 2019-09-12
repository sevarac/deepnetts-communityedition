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


import deepnetts.data.DataSets;
import deepnetts.eval.Evaluators;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import java.util.Map;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.eval.EvaluationMetrics;
import visrec.ri.ml.classification.BinaryClassifierNetwork;

/**
 * Spam  Classification Problem. This example is using  activation in
 * output layer and Cross Entropy Loss function. 
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class CrediCardFraud {

    public static void main(String[] args) throws DeepNettsException, IOException {
        
        int numInputs= 29;
        int numOutputs = 1;
        boolean hasHeader = true;

        // load spam data  set from csv file
        DataSet dataSet = DataSets.readCsv("d:\\datasets\\creditcardfraud\\CreditCardFraudSredjen.csv", numInputs, numOutputs, hasHeader);
        
        // scale data to [0, 1] range
        DataSets.normalizeMax(dataSet);
        
        // split data into training and test set
        DataSet[] trainTestSet = dataSet.split(0.6);
        DataSet trainingSet = trainTestSet[0];
        DataSet testSet = trainTestSet[1];
        
        // create instance of feed forward neural network using its builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)                           // size of the input layer corresponds to number of inputs
                .addFullyConnectedLayer(40, ActivationType.TANH) 
                .addOutputLayer(numOutputs, ActivationType.SIGMOID) // size of output layer corresponds to number of outputs, which is 1 for binary classification problems, and sigmoid transfer function is used for binary classification
                .lossFunction(LossType.CROSS_ENTROPY) // cross entropy loss function is commonly used for classification problems
                .build();
  
        // set parameters of the training algorithm
        neuralNet.getTrainer().setMaxError(0.03f)
                              .setMaxEpochs(10000)
                              .setLearningRate(0.001f);
     
        neuralNet.train(trainingSet);
        
        // test neural network based classifier
        EvaluationMetrics em = Evaluators.evaluateClassifier(neuralNet, testSet);
        System.out.println(em);
        
        
        // Example usage of the trained network
        BinaryClassifierNetwork bnc = new BinaryClassifierNetwork(neuralNet);
        
        float[] testTransaction = getTestTransaction();
        Float result = bnc.classify(testTransaction);
        System.out.println("Fraud probability: "+result);
        
    }
    
    static float[] getTestTransaction() {
        float[] emailFeatures = new float[57];
        emailFeatures[56] = 1;
        return emailFeatures;
    }
    
}
