package deepnetts.net.layers.activation;

import java.io.Serializable;

/**
 * Linear activation function and its derivative
 * 
 * y = x
 * y' = 1
 * 
 * @author zoran
 */
public final class Linear implements ActivationFunction, Serializable {

    @Override
    public float getValue(final float x) {
        return x;
    }

    @Override
    public float getPrime(final float y) {
        return 1;
    }
    
}