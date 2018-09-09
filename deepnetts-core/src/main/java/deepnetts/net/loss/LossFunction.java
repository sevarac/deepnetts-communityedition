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
package deepnetts.net.loss;

/**
 * Interface for all loss functions.
 *
 * TODO: add method to return firts derivative by y : dE/dy ?
 *
 * @see MeanSquaredErrorLoss
 * @see CrossEntropyLoss
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public interface LossFunction {

    /**
     * Calculates pattern error for specified actual and target outputs, adds
     * the error to total error, and returns the pattern error.
     *
     * @param actual actual network output
     * @param target target network output
     * @return error vector
     */
    public float[] addPatternError(float[] actual, float[] target);

    /**
     * Returns the total error value calculated by this loss function.
     *
     * @return total error calculated by this loss function
     */
    public float getTotalValue();

    /**
     * Resets the total error and pattern counter.
     */
    public void reset();

}
