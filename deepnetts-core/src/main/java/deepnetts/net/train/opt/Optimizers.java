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
package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class Optimizers {

    public static final float EPS = 1e-6f; // used for ada grad for numerical stability    should be 1e-8,   1e-7 dl book pg. 308
    public static final float GAMA = 0.9f; // for ada delta

    public static final float sgd(final float learningRate, final float gradient) {
        return -learningRate * gradient;
    }

    public static final float momentum(final float learningRate, final float gradient, final float momentum, final float prevDeltaWeight) {
        return -learningRate * gradient + momentum * prevDeltaWeight;
    }

    public static final float adaGrad(final float learningRate, final float gradient, final float prevGradSqrSum) {
        return -(learningRate / (EPS + (float) Math.sqrt(prevGradSqrSum))) * gradient;
    }

    public static final float rmsProp(final float learningRate, final float gradient, final float prevGradSqrSum) {
        return -(learningRate / (EPS + (float) Math.sqrt(prevGradSqrSum))) * gradient;
    }

    // ada delta treba da ima i prethodnu sumu delta weight - moving average, exponential decaying
    public static final float adaDelta(final float gradient, final float prevGradSqrSum, final float prevDeltaWeightSqrSum) {
        return -(prevDeltaWeightSqrSum / (EPS + (float) Math.sqrt(prevGradSqrSum))) * gradient;
    }

    // adamkoristi sumu kvadrata prethodnih gradijenata ali i samo moving average of prev gradients kao momentum
    // umesto konkretnog gradijenta za update koristi prosecnu sumu prethodnih gradijenata
    public static final float adam() {
        return 0;
    }

    public static final float nestorov() {
        /// makes changes in two iterations first momentum cnange then correction based on gradient
        return 0;
    }

    // da paremtri budu float ...
    public static final float optimize(int type, AbstractLayer layer) {
        switch (type) {

        }
        return 0;
    }

}
