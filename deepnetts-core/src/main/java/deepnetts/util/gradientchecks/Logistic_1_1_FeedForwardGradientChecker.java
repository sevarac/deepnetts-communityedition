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
package deepnetts.util.gradientchecks;

import deepnetts.data.BasicDataSet;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.util.Tensor;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Ovaj mi radi odlicno!!!
 * 
 *
 * http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
 * http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/
 *
 * Use relative error for the comparison
 * http://cs231n.github.io/neural-networks-3/#gradcheck
 *
 * GOALS:
 * at least 4 significant digits (and often many more)
 * 
 * 
 * relative error > 1e-2 usually means the gradient is probably wrong 1e-2 >
 * relative error > 1e-4 should make you feel uncomfortable 1e-4 > relative
 * error is usually okay for objectives with kinks. But if there are no kinks
 * (e.g. use of tanh nonlinearities and softmax), then 1e-4 is too high. 
 * 1e-7 and less you should be happy.
 *
 * h = 1e-4 or 1e-6 (ako je suvise mali uci cu u precision problem) relative err
 * idealno < 1e-7 treba da dude manja od 1e-4 Float isto ume da bude uzrok
 * greske
 *
 * FIX: Primetio sam da prva 4 u svakom lejeru nisu dobra! i to da bas dosta
 * mase. Moguce da ima veze sto je i broj ulaza 4! TODO: Sigurano je greska u
 * implementaciji. Testiraj i jednostavniju mrezu i data set.
 *
 *
 * @author Zoran Sevarac
 * <zoran.sevarac@deepnetts.com>
 */
public class Logistic_1_1_FeedForwardGradientChecker {


    public static void main(String[] args) {
       FeedForwardGradientChecker ffgc =  new FeedForwardGradientChecker (createNetwork(), createDataSet());
       ffgc.run();
    }

    private static FeedForwardNetwork createNetwork() {
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                .addInputLayer(1)           //
                .addOutputLayer(1, ActivationType.SIGMOID)
                .lossFunction(LossType.CROSS_ENTROPY)
                .randomSeed(123)
                .build();

        return neuralNet;
    }

    //creates linear data set
    private static DataSet createDataSet() {
        try {
            DataSet dataSet = BasicDataSet.fromCsv("linear11.csv", 1, 1);
            return dataSet;
        } catch (IOException ex) {
            Logger.getLogger(Logistic_1_1_FeedForwardGradientChecker.class.getName()).log(Level.SEVERE, null, ex);
        }
        return null;
    }

}
