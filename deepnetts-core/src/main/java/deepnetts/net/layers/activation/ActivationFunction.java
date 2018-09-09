package deepnetts.net.layers.activation;

import deepnetts.util.DeepNettsException;

public interface ActivationFunction {

    /**
     * Returns the value of activation function for specified input x
     * @param x input for activation
     * @return 
     */
    public float getValue(float x); // apply(float x)

    public float getPrime(float y);

    
    
    /**
     * Creates and returns specified type of activation function
     * @param type
     * @return 
     */
    public static ActivationFunction create(ActivationType type) {
        switch (type) {
            case LINEAR:
                return new Linear();
            case RELU:
                return new Relu();
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
