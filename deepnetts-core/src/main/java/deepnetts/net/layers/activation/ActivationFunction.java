package deepnetts.net.layers.activation;

import deepnetts.util.DeepNettsException;

/**
 * Common base interface for all activation functions used in layers.
 * Classes implementing this interface should provide methods for calculating
 * value and first derivative of the activation function.
 * Activation function performs non-linear transformation of its input
 * before its sent to layer output.
 * First derivative of a function shows how fast and in what direction function
 * is changing if its input changes, and it is used by training algorithm.
 *
 * For more see <a href="https://en.wikipedia.org/wiki/Activation_function">https://en.wikipedia.org/wiki/Activation_function</a>
 *
 * @see ActivationType
 * @author Zoran Sevarac
 */
public interface ActivationFunction {

    /**
     * Returns the value of activation function for specified input x
     * @param x input for activation
     * @return value of activation function
     */
    public float getValue(float x);


    /**
     * Returns the first derivative of activation function for specified output y
     * @param y output of activation function
     * @return first derivative of activation function
     */
    public float getPrime(float y);


    /**
     * Creates and returns specified type of activation function.
     * A factory method for creating activation functions;
     *
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

}
