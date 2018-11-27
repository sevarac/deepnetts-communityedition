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

package deepnetts.net.layers.activation;

import deepnetts.util.DeepNettsException;

/**
 * Interface for all activaton functions used in layers.
 * 
 * @see https://en.wikipedia.org/wiki/Activation_function
 * @author zoran
 */
public interface ActivationFunction {

    /**
     * Returns the value of activation function for specified input x
     * @param x input for activation
     * @return value of activation function
     */
    public float getValue(float x); // apply(float x)

    
    /**
     * Returns the first derivative of activation function for specified output y
     * @param y output of activation function
     * @return first derivative of activation function
     */
    public float getPrime(float y);

    
    
    /**
     * Creates and returns specified type of activation function.
     * @param type type of the activation function
     * 
     * @return returns instance of specified activation function type
     */
    public static ActivationFunction create(ActivationType type) {
        switch (type) {
            case LINEAR:
                return new Linear();
            case RELU:
                return new Relu();
            case LEAKY_RELU:
                return new LeakyRelu();                
            case SIGMOID:
                return new Sigmoid();
            case TANH:
                return new Tanh();
            default:
                throw new DeepNettsException("Unknown activation function:" + type);
        }
    }

    public static ActivationType LINEAR = ActivationType.LINEAR;
    public static ActivationType RELU = ActivationType.RELU;
    public static ActivationType SIGMOID = ActivationType.SIGMOID;
    public static ActivationType TANH = ActivationType.TANH;

}
