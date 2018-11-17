package deepnetts.net.layers.activation;

import java.io.Serializable;

/**
 * Rectified Linear Activation and its Derivative.
 * 
 * y =  x for x > 0, 
 *      0.1 * x for x<0
 *        - 
 *       | 1, x > 0
 * y' = <
 *       | 0, 0x<=0 
 *        -
 * 
 * @author Zoran Sevarac
 */
public final class LeakyRelu implements ActivationFunction, Serializable {

    @Override
    public float getValue(final float x) {
        return ( x >= 0 ? x : 0.1f*x );  
    }

    @Override
    public float getPrime(final float y) {
        throw new UnsupportedOperationException("Nije implementirana");
//  return ( y > 0 ? 1 : 0);
    }
    
}
