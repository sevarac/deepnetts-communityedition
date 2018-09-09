package deepnetts.net.layers.activation;

import java.io.Serializable;

/**
 *
 * @author zoran
 */
public final class Relu implements ActivationFunction, Serializable {

    @Override
    public float getValue(final float x) {
        return Math.max(0, x);  
    }

    @Override
    public float getPrime(final float y) {
         return ( y > 0 ? 1 : 0);
    }
    
}
