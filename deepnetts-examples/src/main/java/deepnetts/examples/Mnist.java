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
 * this program. If not, see <https://www.gnu.org/licenses/>.
 */
package deepnetts.examples;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.DeepNettsException;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.ConfusionMatrix;
import javax.visrec.ml.eval.EvaluationMetrics;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Example of training Convolutional network for MNIST data set. Note: in order
 * to run this example you must download mnist data set and update image paths
 * in train.txt file
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class Mnist {

    // input image dimensions
    final int IMAGE_WIDTH = 28;
    final int IMAGE_HEIGHT = 28;

    // data set path and training files
    final String DATA_SET_PATH = "D:/datasets/mnist/train";
    final String LABELS_FILE = DATA_SET_PATH + "/labels.txt";
    final String TRAINING_FILE = DATA_SET_PATH + "/train.txt";

    static final Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    public void run() throws DeepNettsException, IOException {

        LOGGER.info("Training convolutional network with MNIST data set");
        LOGGER.info("Creating image data set...");

        // create a data set from images and labels
        ImageSet imageSet = new ImageSet(IMAGE_WIDTH, IMAGE_HEIGHT);
        imageSet.setInvertImages(true);
        imageSet.loadLabels(new File(LABELS_FILE));
        imageSet.loadImages(new File(TRAINING_FILE), 1000);
        //  imageSet.zeroMean();
        imageSet.countByClasses();      
        
        ImageSet[] imageSets = imageSet.split(0.7, 0.3);
        int labelsCount = imageSet.getLabelsCount();

        LOGGER.info("------------------------------------------------");
        LOGGER.info("CREATING NEURAL NETWORK");
        LOGGER.info("------------------------------------------------");

        // create convolutional neural network architecture
        ConvolutionalNetwork neuralNet = ConvolutionalNetwork.builder()
                .addInputLayer(IMAGE_WIDTH, IMAGE_HEIGHT, 3)
                .addConvolutionalLayer(3, 3, 12)
                .addMaxPoolingLayer(2, 2)
                .addFullyConnectedLayer(30)
                .addOutputLayer(labelsCount, ActivationType.SOFTMAX)
                .hiddenActivationFunction(ActivationType.TANH)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();
        
        LOGGER.info(neuralNet);        
        
        // create a trainer and train network
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
        trainer.setLearningRate(0.01f)
                .setMaxError(0.05f);
        
        trainer.train(imageSets[0]);

        // Test trained network
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        evaluator.evaluate(neuralNet, imageSets[1]);
        LOGGER.info("------------------------------------------------");
        LOGGER.info("Classification performance measure" + System.lineSeparator());
        LOGGER.info("TOTAL AVERAGE");
        LOGGER.info(evaluator.getTotalAverage());
        LOGGER.info("By Class");
        Map<String, EvaluationMetrics> byClass = evaluator.getPerformanceByClass();
        byClass.entrySet().stream().forEach((entry) -> {
            LOGGER.info("Class " + entry.getKey() + ":");
            LOGGER.info(entry.getValue());
            LOGGER.info("----------------");
        });
        
         LOGGER.info("CONFUSION MATRIX");
        ConfusionMatrix cm = evaluator.getConfusionMatrix();
        LOGGER.info(cm);

        // Save network to file
        FileIO.writeToFile(neuralNet, "mnistDemo.dnet");
    }

    public static void main(String[] args) throws IOException {
        (new Mnist()).run();
    }


}
