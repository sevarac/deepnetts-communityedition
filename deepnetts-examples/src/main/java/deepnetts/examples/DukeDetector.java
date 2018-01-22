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

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.OptimizerType;
import deepnetts.eval.ConvolutionalClassifierEvaluator;
import deepnetts.eval.PerformanceMeasure;
import deepnetts.net.loss.LossType;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Convolutional Neural Network that learns to detect Duke images.
 * Example how to create and train convolutional network for image classification.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class DukeDetector {

     static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());   

    public static void main(String[] args) throws FileNotFoundException, IOException {
        int imageWidth = 64;
        int imageHeight = 64;

        String trainingFile = "D:\\datasets\\DukeSet\\train.txt";        
        String labelsFile = "D:\\datasets\\DukeSet\\labels.txt";
//        String labelsFile = "datasets/DukeSet/labels.txt";
//        String trainingFile = "datasets/DukeSet/train.txt";
  //      String testFile = "datasets/DukeSet/test.txt";

        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);

        LOGGER.info("Loading images...");

        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile), true);
        imageSet.invert();
       // imageSet.zeroMean();
        imageSet.shuffle();

        LOGGER.info("Creating neural network...");

        ConvolutionalNetwork convNet = new ConvolutionalNetwork.Builder()
                .addInputLayer(imageWidth, imageHeight, 3)
                .addConvolutionalLayer(5, 5, 1, ActivationType.TANH)
                .addMaxPoolingLayer(2, 2, 2)
                .addFullyConnectedLayer(3, ActivationType.TANH)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .withLossFunction(LossType.CROSS_ENTROPY)
                .build();

        convNet.setOutputLabels(imageSet.getLabels());

        LOGGER.info("Training neural network");

        // create a set of convolutional networks and do training, crossvalidation and performance evaluation
        BackpropagationTrainer trainer = new BackpropagationTrainer(convNet);
        trainer.setMaxError(0.5f)
               .setLearningRate(0.01f)
               .setOptimizer(OptimizerType.SGD)
               .setMomentum(0.2f);
        trainer.train(imageSet);

        // to save neural network to file on disk
        FileIO.writeToFile(convNet, "DukeDetector.dnet");

        // to load neural network from file use FileIO.createFromFile
        // dukeNet = FileIO.createFromFile("DukeDetector.dnet");  
        // to serialize network in json use FileIO.toJson    
        // System.out.println(FileIO.toJson(dukeNet));
        
        // to evaluate recognizer with image set
        ConvolutionalClassifierEvaluator evaluator = new ConvolutionalClassifierEvaluator();
        Map<String, PerformanceMeasure>  pm =  evaluator.evaluate(convNet, imageSet);
        System.out.println(pm);

        // to use recognizer for single image
//        BufferedImage image = ImageIO.read(new File("/home/zoran/datasets/DukeSet/duke/duke7.jpg"));
//        DeepNettsImageClassifier imageClassifier = new DeepNettsImageClassifier(convNet);
//        ClassificationResults<ClassificationResult> results = imageClassifier.classify(image);

     //   System.out.println(results.toString());
    }

}