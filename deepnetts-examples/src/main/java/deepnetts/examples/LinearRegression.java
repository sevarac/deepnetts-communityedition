/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */    

package deepnetts.examples;

import deepnetts.data.BasicDataSet;
import deepnetts.data.DataSet;
import deepnetts.data.DataSets;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.Tensor;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Minimal example for linear regression using FeedForwardNetwork.
 * Fits a straight line through the data.
 Uses a single addLayer with one output and linear activation function, and Mean Squared Error for Loss function.
 You can use linear regression to roughly estimate a global trend in data.
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class LinearRegression {
    
    public static void main(String[] args) throws IOException {
              
            int inputsNum = 1;
            int outputsNum = 1;
            String csvFilename = "linear.csv";
                      
            // load and create data set from csv file
            DataSet dataSet = DataSets.readCsv(csvFilename , inputsNum, outputsNum);
            
            // create neural network using network specific builder
            FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                    .addInputLayer(inputsNum)
             //       .addDenseLayer(10, ActivationType.TANH)
                    .addOutputLayer(outputsNum, ActivationType.LINEAR)
                    .lossFunction(LossType.MEAN_SQUARED_ERROR)
                    .build();
            
//            neuralNet.getTrainer()
//                    .setMaxError(0.0002f)
//                    .setMaxEpochs(10000)
//                    .setBatchMode(true)
//                    .setLearningRate(0.001f);
            
            // train network using loaded data set
            neuralNet.train(dataSet);
                   
            // learned model
            float slope = neuralNet.getLayers().get(1).getWeights().get(0);
            float intercept = neuralNet.getLayers().get(1).getBiases()[0];
            System.out.println("Original function: y = 0.5 * x + 0.2");
            System.out.println("Estimated/learned function: y = "+slope+" * x + "+intercept);

            // perform prediction for some input value
            neuralNet.setInput(Tensor.create(1, 1, new float[] {0.2f}));
            System.out.println("Predicted output for 0.2 :" + Arrays.toString(neuralNet.getOutput()));
            
            // plot predictions for some random data 
            plotPredictions(neuralNet);
    }
    
    
    public static void plotPredictions(FeedForwardNetwork nnet) {
        double[][] data = new double[100][2];
        
        for(int i=0; i<data.length; i++) {
            data[i][0] =  Math.random();
            nnet.setInput(new Tensor(1, 1, new float[] { (float)data[i][0] }));            
            data[i][1] = nnet.getOutput()[0];
        }        
        
        Plot.scatter(data);   
    }
    
    
    
}
