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
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.DeepNettsException;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.EvaluationMetrics;
import deepnetts.net.loss.LossType;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class Clouds {

    int imageWidth = 64; //125
    int imageHeight = 64; // 125

    //String labelsFile = "/home/zoran/datasets/cifar10/train/labels.txt";
    String labelsFile = "D:\\datasets\\clouds\\swimcat\\labels.txt";
    //String trainingFile = "datasets/cifar10/train.txt";
    //String trainingFile = "/home/zoran/datasets/cifar10/train/train.txt";
    String trainingFile = "D:\\datasets\\clouds\\swimcat\\index.txt";
   // String testFile = "datasets/cifar10/test.txt";

    static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    public void run() throws DeepNettsException, IOException {
        LOGGER.info("Loading images...");
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile));
//        imageSet.invert();
    //    imageSet.zeroMean();
        imageSet.shuffle();

        int labelsCount = imageSet.getLabelsCount();

        ImageSet[] imageSets = imageSet.split(0.7, 0.3);

        LOGGER.info("Creating neural network...");

        ConvolutionalNetwork neuralNet = ConvolutionalNetwork.builder()
                                        .addInputLayer(imageWidth, imageHeight)
                                        .addConvolutionalLayer(3, 3, 8)
                                        .addMaxPoolingLayer(2, 2, 2)
                                        .addConvolutionalLayer(3, 3, 16)
                                        .addMaxPoolingLayer(2, 2, 2)
////                                        .addConvolutionalLayer(3, 3, 12)
////                                        .addMaxPoolingLayer(2, 2, 2)
////                                     //   .addFullyConnectedLayer(30)
//                                        .addFullyConnectedLayer(16)
                                        .addFullyConnectedLayer(16)
                                        .addFullyConnectedLayer(9)
                                        .addOutputLayer(labelsCount, ActivationType.SOFTMAX)
                                        .hiddenActivationFunction(ActivationType.TANH)
                                        .lossFunction(LossType.CROSS_ENTROPY)
                                        .build();

        
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
        trainer.setLearningRate(0.005f);
        trainer.setMaxError(0.1f);
        trainer.setMaxEpochs(1000);
        trainer.setMomentum(0.9f);
        trainer.setOptimizer(OptimizerType.SGD);   // sa momentumom raste greska!!!
        trainer.train(imageSets[0]);

        // Test trained network
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        EvaluationMetrics pm = evaluator.evaluate(neuralNet, imageSets[1]);
        LOGGER.info("------------------------------------------------");
        LOGGER.info("Classification performance measure"+System.lineSeparator());
        LOGGER.info("TOTAL AVERAGE");
        LOGGER.info(evaluator.getTotalAverage());
        LOGGER.info("By Class");
        Map<String, EvaluationMetrics>  byClass = evaluator.getPerformanceByClass();
        byClass.entrySet().stream().forEach((entry) -> {
                                        LOGGER.info("Class " + entry.getKey() + ":");
                                        LOGGER.info(entry.getValue());
                                        LOGGER.info("----------------");
                                    });

    }

    public static void main(String[] args) throws DeepNettsException, IOException {
            (new Clouds()).run();
    }
}