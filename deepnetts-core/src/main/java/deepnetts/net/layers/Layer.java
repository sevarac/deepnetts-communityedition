package deepnetts.net.layers;

import deepnetts.util.Tensor;

/**
 * Common base interface for all types of neural network layers.
 * Layer is a basic building block of a neural network, and neural network
 * typically consists of a sequence of layers.
 *
 * @see AbstractLayer
 * @author Zoran Sevarac
 */
public interface Layer {

    /**
     * Performs layer calculation in forward pass of a neural network.
     */
    public void forward();

    /**
     * Performs weight parameters adjustment in backward pass during training of a neural network.
     */
    public void backward();

    /**
     * Returns layer outputs (as a tensor).
     * @return layer outputs tensor
     */
    public Tensor getOutputs();

    /**
     * Returns layer deltas (as a tensor).
     * Deltas are accumulated errors propagated from the next layer.
     * @return layer deltas tensor
     */
    public Tensor getDeltas();


//    public Tensor getWeights();

}