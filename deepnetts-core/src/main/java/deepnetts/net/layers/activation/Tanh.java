package deepnetts.net.layers.activation;

import java.io.Serializable;

/**
 * Hyperbolic tangens activation function
 * 
 * @author zoran
 */
public final class Tanh implements ActivationFunction, Serializable {

    @Override
    public float getValue(final float x) {
       final float e2x = (float)Math.exp(2*x);   
       return (e2x-1) / (e2x+1); // calculate tanh 
    }

    @Override
    public float getPrime(final float y) {
        return (1-y*y);
    }
    
}
