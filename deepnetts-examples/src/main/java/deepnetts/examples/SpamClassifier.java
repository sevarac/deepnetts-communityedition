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
import deepnetts.data.MLDataItem;
import deepnetts.data.norm.MaxNormalizer;
import deepnetts.eval.Evaluators;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.DeepNettsException;
import java.io.IOException;
import javax.visrec.ml.classification.BinaryClassifier;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ri.ml.classification.FeedForwardNetBinaryClassifier;

/**
 * Spam  Classification example.
 * Minimal example how to train a binary classifier for spam email  classification, using Feed Forward neural network.
 * Data is given as CSV file.
 *
 * <Link to tutorial>
 * additional links to related  tutorials
 * 
 * Link to original data set
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class SpamClassifier {

    public static void main(String[] args) throws DeepNettsException, IOException {

        int numInputs = 57;
        int numOutputs = 1;
        
        // load spam data  set from csv file - what is csv file?
        DataSet dataSet = DataSets.readCsv("datasets/spam.csv", numInputs, numOutputs, true);             

        // split data set into train and test set (- link to why splitting data into training and test set? what is basic machine learning workflow)
        DataSet<MLDataItem>[] trainAndTestSet = dataSet.split(0.6, 0.4);
        DataSet<MLDataItem> trainingSet = trainAndTestSet[0];
        DataSet<MLDataItem> testSet = trainAndTestSet[1];
                
        // normalize data - Why normalize data? and learn more about common data preprocessing you need 
        MaxNormalizer norm = new MaxNormalizer(trainingSet);    //
        norm.normalize(trainingSet);
        norm.normalize(testSet);
        
        // create instance of feed forward neural network using its builder
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(numInputs)
                .addFullyConnectedLayer(25, ActivationType.TANH)
                .addOutputLayer(numOutputs, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        // set training settings - how to set training parameters?
        neuralNet.getTrainer().setMaxError(0.2f)
                              .setLearningRate(0.01f);
        
        // run training
        neuralNet.train(trainingSet);
        
        // test network /  evaluate classifier - how to interpret results of classifier evaluation
        EvaluationMetrics em = Evaluators.evaluateClassifier(neuralNet, testSet);
        System.out.println(em);
        
        // using trained network: create binary classifier using trained network
        BinaryClassifier<float[]> binClassifier = new FeedForwardNetBinaryClassifier(neuralNet);
        
        // get single feature array from test set
        float[] testEmail = trainAndTestSet[1].get(0).getInput().getValues();
        // feed the classifer and get result - spam probability
        Float result = binClassifier.classify(testEmail);        
        System.out.println("Spam probability: "+result);        
    }
    

}