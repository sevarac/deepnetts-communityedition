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
import deepnetts.eval.Evaluators;
import deepnetts.eval.EvaluationMetrics;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import java.io.IOException;
import java.util.Arrays;

/**
 * Minimal example for logistic regression using FeedForwardNetwork.
 * Can perform binary classification (single output true/false)
 Uses only input and output addLayer with sigmoid activation function
 Just specify number of inputs and provide data set
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class LogisticRegression {

    public static void main(String[] args) throws IOException {

        BasicDataSet dataSet = DataSets.readCsv("datasets/sonar.csv", 60, 1); // get data from some file or method
        DataSet[] trainTestSet = DataSets.trainTestSplit(dataSet, 0.7);

        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(60)
                .addOutputLayer(1, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .build();

        BackpropagationTrainer trainer = neuralNet.getTrainer();
        trainer.setLearningRate(0.01f)
               .setMaxError(0.3f)
                .setMaxEpochs(8000);

        neuralNet.train(trainTestSet[0]);

        EvaluationMetrics pm = neuralNet.test(trainTestSet[1]);
        System.out.println(pm);

        float[] predictedOut = neuralNet.predict(new float[]{0.0303f,0.0353f,0.0490f,0.0608f,0.0167f,0.1354f,0.1465f,0.1123f,0.1945f,0.2354f,0.2898f,0.2812f,0.1578f,0.0273f,0.0673f,0.1444f,0.2070f,0.2645f,0.2828f,0.4293f,0.5685f,0.6990f,0.7246f,0.7622f,0.9242f,1.0000f,0.9979f,0.8297f,0.7032f,0.7141f,0.6893f,0.4961f,0.2584f,0.0969f,0.0776f,0.0364f,0.1572f,0.1823f,0.1349f,0.0849f,0.0492f,0.1367f,0.1552f,0.1548f,0.1319f,0.0985f,0.1258f,0.0954f,0.0489f,0.0241f,0.0042f,0.0086f,0.0046f,0.0126f,0.0036f,0.0035f,0.0034f,0.0079f,0.0036f,0.0048f});
        //float[] predictedOut = neuralNet.predict(new float[]{0.0762f,0.0666f,0.0481f,0.0394f,0.0590f,0.0649f,0.1209f,0.2467f,0.3564f,0.4459f,0.4152f,0.3952f,0.4256f,0.4135f,0.4528f,0.5326f,0.7306f,0.6193f,0.2032f,0.4636f,0.4148f,0.4292f,0.5730f,0.5399f,0.3161f,0.2285f,0.6995f,1.0000f,0.7262f,0.4724f,0.5103f,0.5459f,0.2881f,0.0981f,0.1951f,0.4181f,0.4604f,0.3217f,0.2828f,0.2430f,0.1979f,0.2444f,0.1847f,0.0841f,0.0692f,0.0528f,0.0357f,0.0085f,0.0230f,0.0046f,0.0156f,0.0031f,0.0054f,0.0105f,0.0110f,0.0015f,0.0072f,0.0048f,0.0107f,0.0094f});
        System.out.println(Arrays.toString(predictedOut));

    }

}
