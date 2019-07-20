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

package deepnetts.eval;

/**
 * This class calculates values used for evaluation metrics for regression problems.
 *
 * @author Zoran Sevarac
 */
public class MeanSquaredError {

    private float squaredSum;
    private int patternCount;

    public void add(float[] predicted, float[] target) {
        for(int i=0; i<predicted.length; i++)
            squaredSum += Math.pow((predicted[i] - target[i]), 2);

        patternCount++;
    }

    /**
     * Returns squared error sum (RSS, or residual square sum)
     * @return
     */
    public float getSquaredSum() {
        return squaredSum;
    }

    /**
     * Returns mean squared error
     * @return
     */
    public float getMeanSquaredSum() {
        return squaredSum / patternCount;
    }

    public float getRootMeanSquaredSum() {
        return (float)Math.sqrt(squaredSum / patternCount);
    }

}