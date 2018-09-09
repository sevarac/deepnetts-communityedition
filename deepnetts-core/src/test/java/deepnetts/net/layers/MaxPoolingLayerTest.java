package deepnetts.net.layers;

import deepnetts.net.layers.activation.ActivationType;
import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Ignore;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class MaxPoolingLayerTest {

    /**
     * Test of forward method, of class MaxPoolingLayer.
     */
    @Test
    public void testForwardSingleChannel() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        Tensor filter = new Tensor(3, 3,
                new float[]{
                             0.1f, 0.2f, 0.3f,
                            -0.11f, -0.2f, -0.3f,
                             0.4f, 0.5f, 0.21f
                           });

        // set biases to zero
        float[] biases = new float[]{0.0f};

        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 1);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = filter;
        convLayer.biases = biases;

        inputLayer.setInput(input);
        convLayer.forward();    
        
        /* Output: 
           [-0.40289998, -0.24970004, 0.11339998, 0.072799996, 0.2441,      0.38160002,
             0.20070001,  0.45139998, 0.5405,     0.52190006,  0.4957,      0.4742, 
             0.2084,      0.4037,     0.39240003, 0.1401,     -0.08989998, -0.066199996,
             0.27409998,  0.45,       0.72080004, 0.99470013,  0.77730006,  0.52349997,
             0.2044,      0.4385,     0.29759997, 0.1762,      0.074000016, 0.23410001,
            -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002,  0.0064999983] */

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(3, 3,
                new float[]{
                                0.45139998f, 0.5405f, 0.4957f,
                                0.45f, 0.99470013f, 0.77730006f,
                                0.4385f, 0.29759997f, 0.23410001f
                });

        /* maxIdxs  1,1     1,2     1,4
                    3,1     3,3     3,4
                    4,1     4,2     4,5  */
        
        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-8f);
    }

    @Test
    public void testForwardMultipleChannels() {

        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{
                    0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 2);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        
        convLayer.filters[0] = new Tensor(3, 3,
                new float[]{
                    0.1f, 0.2f, 0.3f,
                   -0.11f, -0.2f, -0.3f,
                    0.4f, 0.5f, 0.21f
                });
        
        convLayer.filters[1] = new Tensor(3, 3,
                new float[]{
                    0.11f, 0.21f, 0.31f,
                   -0.21f, -0.22f, -0.23f,
                    0.31f, 0.31f, 0.31f
                });
        
        // set biases to zero
        convLayer.biases = new float[]{0.0f, 0.0f};

        inputLayer.setInput(input);
        convLayer.forward();    // output from convolutional layer:
        /*         
                                    -0.40289998, -0.24970004, 0.11339998, 0.072799996, 0.2441,      0.38160002,
                                     0.20070001,  0.45139998, 0.5405,     0.52190006,  0.4957,      0.4742,
                                     0.2084,      0.4037,     0.39240003, 0.1401,     -0.08989998, -0.066199996,
                                     0.27409998,  0.45,       0.72080004, 0.99470013,  0.77730006,  0.52349997,
                                     0.2044,      0.4385,     0.29759997, 0.1762,      0.074000016, 0.23410001,
                                    -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002,  0.0064999983,
                                    
                                    -0.20889999, -0.26760003, -0.010199998, -6.999895E-4,  0.22350001,   0.22450002,
                                     0.3319,      0.48950002,  0.44680002,   0.4791,       0.40600002,   0.25570002, 
                                     0.19569999,  0.3932,      0.2622,       0.014099985, -0.060699996, -0.15130001, 
                                     0.2328,      0.3976,      0.6252,       0.6627,       0.8222,       0.4177, 
                                     0.27,        0.31350002,  0.23630002,  -0.0035999827, 0.04750003,   0.10620001, 
                                     0.028599992, 0.105699986, 0.18150005,   0.033699997,  0.064200014, -0.014600009"        
         */

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(3, 3, 2,
                new float[]{
                    0.45139998f, 0.5405f, 0.4957f,
                    0.45f, 0.99470013f, 0.77730006f,
                    0.4385f, 0.29759997f, 0.23410001f,
                    0.48950002f, 0.4791f, 0.40600002f,
                    0.3976f, 0.6627f, 0.8222f,
                    0.31350002f, 0.23630002f, 0.10620001f
                });

        /* maxIdxs  1,1     1,2     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5 
        
                            1,1     1,3     1,4        
                            3,1     3,3     3,4 
                            4,1     4,2     4,5
         */
        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-8f);
    }

    /**
     * Test of backward method, of class MaxPoolingLayer. propusti gresku iz
     * sledeceg lejera daltu unazad samo za neurone koji su bili max (na osnovu
     * zapamcenih pozicija)
     */
    @Test
    public void testBackwardFromFullyConnectedToSingleChannel() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        Tensor filter = new Tensor(3, 3,
                new float[]{0.1f, 0.2f, 0.3f,
                    -0.11f, -0.2f, -0.3f,
                    0.4f, 0.5f, 0.21f});

        // set biases to zero
        float[] biases = new float[]{0.0f};

        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 1);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = filter;
        convLayer.biases = biases;

        inputLayer.setInput(input);
        convLayer.forward();    // vidi koliki je output i njega onda pooluj

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        /* Max Pooling Output
                                new float[] {0.45139998f, 0.5405f, 0.4957f,
                                             0.45f, 0.99470013f, 0.77730006f,
                                             0.4385f, 0.29759997f, 0.23410001f});        
         */
                /* maxIdxs  1,1     1,2     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5  */
        DenseLayer nextLayer = new DenseLayer(2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.init(); // init weights               
        nextLayer.setDeltas(new Tensor(0.1f, 0.2f));
        // poslednja dimenzija matrice tezina je 2 - koliko ima neurona u fc. - zasto je 3x3x1  X  2  (prev layer x fcCols)
        /* test samo sa jednim neuronom u fc i delta 0.1, pomnozi sve tezine sa 0.1
        
        weights sa dva neurona u fc:
            0.18075174, 0.5545214, 0.072818756,
            0.31912476, -0.49894053, -0.6323205,
            0.2685551, 0.4376064, -0.34319848, 
        
            0.11627263, -0.3802099, 0.60284144,
            0.15107232, -0.5185875, -0.41857186,
            0.7019462, -0.0617525, -0.64165723
        
         */

        instance.backward();
        Tensor actual = instance.getDeltas();

        // sum delta * weight and transpose
        // Test: 0.1 * 0.18075174 + 0.2 * 11627263 = 0.0413297 ... 
        Tensor expected = new Tensor(3, 3,
                new float[]{0.0413297f, 0.062126942f, 0.16724476f, 
                           -0.020589843f, -0.15361156f, 0.03141014f,
                            0.12785016f, -0.14694643f, -0.1626513f});

        assertArrayEquals(actual.getValues(), expected.getValues(), 1e-8f);
    }
    
    @Test
    public void testBackwardMultiChannelFromFullyConnected() {
        RandomGenerator.getDefault().initSeed(123);
        
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                    -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                    -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                    0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                    0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                    0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        Tensor filter = new Tensor(3, 3,
                new float[]{0.1f, 0.2f, 0.3f,
                    -0.11f, -0.2f, -0.3f,
                    0.4f, 0.5f, 0.21f});

        // set biases to zero
        float[] biases = new float[]{0.0f, 0.0f};

        ConvolutionalLayer convLayer = new ConvolutionalLayer(3, 3, 2);
        convLayer.setPrevLayer(inputLayer);
        convLayer.activationType = ActivationType.LINEAR;
        convLayer.init();
        convLayer.filters[0] = filter; // treba mi 2 filtera
        convLayer.filters[1] = filter;
        convLayer.biases = biases;

        inputLayer.setInput(input);
        convLayer.forward();    // vidi koliki je output i njega onda pooluj

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        /* Max Pooling Output
                                new float[] {0.45139998f, 0.5405f, 0.4957f,
                                             0.45f, 0.99470013f, 0.77730006f,
                                             0.4385f, 0.29759997f, 0.23410001f});        
         */
                /* maxIdxs  1,1     1,2     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5  */
        DenseLayer nextLayer = new DenseLayer(2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.init(); // init weights               
        nextLayer.setDeltas(new Tensor(0.1f, 0.2f));
        // poslednja dimenzija matrice tezina je 2 - koliko ima neurona u fc. - zasto je 3x3x1  X  2  (prev layer x fcCols)
        /* test samo sa jednim neuronom u fc i delta 0.1, pomnozi sve tezine sa 0.1
        
        weights sa dva neurona u fc:
            0.0862301, -0.28197122, 0.44707918,
            0.112038255, -0.38459483, -0.31042123,
            0.5205773, -0.04579687, -0.47586578, 
        
            -0.4501995, -0.4715696, -0.4138012,
            -0.44830847, -0.1773395, -0.08274263,
             0.47323978, 0.41012484, 0.19486493, 
        
            0.08227211, -0.54566085, -0.11338204, 
            -1.4930964E-4, -0.26475245, 0.52672565, 
            -0.38034722, 0.3306117, -0.26243588, 
        
            0.31195676, -0.04197538, -0.5319252,
            -0.06671178, -0.29084632, -0.31613958, 
            -0.025327086, 0.12511307, -0.3920598
        
         */

        instance.backward();
        Tensor actual = instance.getDeltas();

        // sum delta * weight and transpose
        // Test: 0.1 * 0.0862301 + 0.2 * 0.08227211 =  0.025077432   +
        // Test: 0.1 * -0.28197122 + 0.2 * -0.54566085 = −0.137329292 +
        // Test: 0.1 * -0.4501995 + 0.2 * 0.31195676 = 0.017371402  
        // Test: 0.1 * 0.19486493 + 0.2 * -0.3920598 = −0.05892546
        
        Tensor expected = new Tensor(3, 3, 2,
                new float[]{ 0.025077432f, 0.011173964f, -0.024011713f,
                            -0.1373293f, -0.091409974f, 0.06154266f,
                             0.022031512f, 0.07430301f, -0.100073755f,
                             
                             0.017371401f, -0.058173206f, 0.04225856f,
                             -0.055552036f, -0.075903215f, 0.0660351f, 
                             -0.14776516f, -0.07150218f, -0.05892546f });

        assertArrayEquals(actual.getValues(), expected.getValues(), 1e-8f);
    }    
    
    
    @Test
    public void testBackwardFromSingleConvolutionalToSinglePoolingChannel() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(12, 12, 1);
        Tensor input = new Tensor(12, 12,     // ovaj input treba povecati jer se ubrzi prepolovi
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f, 0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f, -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f, -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f, 0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f, 0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f, 0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f, -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f, -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f, 0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f, 0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f                             
                });

        Tensor filter = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f
                            });

        float[] biases = new float[]{0.0f};

         
        ConvolutionalLayer prevLayer = new ConvolutionalLayer(3, 3, 1);
        prevLayer.setPrevLayer(inputLayer);        
        prevLayer.activationType = ActivationType.LINEAR;
             
        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(prevLayer);
        prevLayer.setNextlayer(instance);
               
        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 1);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
       
        prevLayer.init();
        instance.init();  
        nextLayer.init();        
        
        prevLayer.filters[0] = filter;
        prevLayer.biases = biases;        
        
        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.filters[0] = filter;
        nextLayer.biases = biases;                
        
        // propagate forward and backward
        inputLayer.setInput(input);
        prevLayer.forward();
        instance.forward();
        nextLayer.forward();
        nextLayer.setDeltas(new Tensor(6, 6,
                                        new float[] { 
                                            0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f,
                                            0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f,
                                            0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
                                            0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f,
                                            0.51f, 0.52f, 0.53f, 0.54f, 0.55f, 0.56f,
                                            0.61f, 0.62f, 0.63f, 0.64f, 0.65f, 0.66f
                                        }));
        instance.backward();
        
        Tensor actual = instance.getDeltas();
                // da li su ove delte ispravno propagirane unazad?
        Tensor expected = new Tensor(6, 6,
                new float[]{
                                0.0376f, 0.087f, 0.0894f, 0.0918f, 0.0942f, 0.0883f,
                                0.1476f, 0.2461f, 0.2596f, 0.2731f, 0.2866f, 0.22479999f,
                                0.24459998f, 0.3811f, 0.3946f, 0.40809998f, 0.4216f, 0.3208f,
                                0.34159997f, 0.5161f, 0.5296f, 0.5431f, 0.5566f, 0.4168f,
                                0.4386f, 0.6511001f, 0.6646f, 0.6781f, 0.69159997f, 0.5128f,
                                0.3216f, 0.3561f, 0.3636f, 0.3711f, 0.3786f, 0.2318f    
                           });
       
        assertArrayEquals(expected.getValues(), actual.getValues(), 1e-7f);
    }             
    
   @Test
    public void testBackwardFromTwoConvolutionalChannelsToSinglePoolingChannel() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(12, 12, 1);
        Tensor input = new Tensor(12, 12,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f, 0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f, -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f, -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f, 0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f, 0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f, 0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f, -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f, -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f, 0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f, 0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f                             
                });

        Tensor filter = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f
                            });
        
        ConvolutionalLayer prevLayer = new ConvolutionalLayer(3, 3, 1);
        prevLayer.setPrevLayer(inputLayer);        
        prevLayer.activationType = ActivationType.LINEAR;
             
        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(prevLayer);
        prevLayer.setNextlayer(instance);
               
        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 2);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
       
        prevLayer.init();
        instance.init();  
        nextLayer.init();        
        
        prevLayer.filters[0] = filter;
        prevLayer.biases = new float[]{0.0f};        
        
        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.filters[0] = filter;
        nextLayer.filters[1] = filter;
        nextLayer.biases = new float[] {0.0f, 0.0f};      
                           
        // propagate forward and backward
        inputLayer.setInput(input);
        prevLayer.forward();
        instance.forward();
        nextLayer.forward();
        nextLayer.setDeltas(new Tensor(6, 6, 2,
                                        new float[] { 
                                            0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f,
                                            0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f,
                                            0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
                                            0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f,
                                            0.51f, 0.52f, 0.53f, 0.54f, 0.55f, 0.56f,
                                            0.61f, 0.62f, 0.63f, 0.64f, 0.65f, 0.66f,
                                            
                                            0.22f,  0.24f,  0.26f,  0.28f,  0.3f,  0.32f,
                                            0.42f,  0.44f,  0.46f,  0.48f,  0.5f,  0.52f,
                                            0.62f,  0.64f,  0.66f,  0.68f,  0.7f,  0.72f,
                                            0.82f,  0.84f,  0.86f,  0.88f,  0.9f,  0.92f,
                                            1.02f,  1.04f,  1.06f,  1.08f,  1.1f,  1.12f,
                                            1.22f,  1.24f,  1.26f,  1.28f,  1.3f,  1.32f                                            
                                        }));
        instance.backward();
        
        Tensor actual = instance.getDeltas();

        Tensor expected = new Tensor(6, 6,
                new float[]{
                                0.11280f,  0.26100f,  0.26820f,  0.27540f,  0.28260f,  0.26490f,
                                0.44280f,  0.73830f,  0.77880f,  0.81930f,  0.85980f,  0.67440f,
                                0.73380f,  1.14330f,  1.18380f,  1.22430f,  1.26480f,  0.96240f,
                                1.02480f,  1.54830f,  1.58880f,  1.62930f,  1.66980f,  1.25040f,
                                1.31580f,  1.95330f,  1.99380f,  2.03430f,  2.07480f,  1.53840f,
                                0.96480f,  1.06830f,  1.09080f,  1.11330f,  1.13580f,  0.69540f
                           });

        assertArrayEquals(expected.getValues(), actual.getValues(), 1e-4f);
    }         
    
    /**
     * Test delta propagation from single convolutional channel back to two max pooling channels
     */
   @Test
    public void testBackwardFromSingleConvolutionalToTwoPoolingChannels() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(12, 12, 1);
        Tensor input = new Tensor(12, 12,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f, 0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f, -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f, -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f, 0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f, 0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f, 0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f, -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f, -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f, 0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f, 0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f                             
                });

        Tensor filter = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f
                            });
        
        Tensor filter2 = new Tensor(3, 3, 2,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f,
                              
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f                              
                            });        
        
        ConvolutionalLayer prevLayer = new ConvolutionalLayer(3, 3, 2);
        prevLayer.setPrevLayer(inputLayer);        
        prevLayer.activationType = ActivationType.LINEAR;
             
        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(prevLayer);
        prevLayer.setNextlayer(instance);
               
        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 1);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
       
        prevLayer.init();
        instance.init();  
        nextLayer.init();        
        
        prevLayer.filters[0] = filter;
        prevLayer.filters[1] = filter;
        prevLayer.biases = new float[] {0.0f, 0.0f};        
        
        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.filters[0] = filter2;
        nextLayer.biases = new float[] {0.0f};     
                           
        // propagate forward and backward
        inputLayer.setInput(input);
        instance.forward();
        nextLayer.forward();
        nextLayer.setDeltas(new Tensor(6, 6,
                                        new float[] { 
                                            0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f,
                                            0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f,
                                            0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
                                            0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f,
                                            0.51f, 0.52f, 0.53f, 0.54f, 0.55f, 0.56f,
                                            0.61f, 0.62f, 0.63f, 0.64f, 0.65f, 0.66f
                                        }));
        instance.backward();
        
        Tensor actual = instance.getDeltas();

        Tensor expected = new Tensor(6, 6, 2,
                new float[]{
                                0.0376f, 0.087f, 0.0894f, 0.0918f, 0.0942f, 0.0883f,
                                0.1476f, 0.2461f, 0.2596f, 0.2731f, 0.2866f, 0.22479999f,
                                0.24459998f, 0.3811f, 0.3946f, 0.40809998f, 0.4216f, 0.3208f,
                                0.34159997f, 0.5161f, 0.5296f, 0.5431f, 0.5566f, 0.4168f,
                                0.4386f, 0.6511001f, 0.6646f, 0.6781f, 0.69159997f, 0.5128f,
                                0.3216f, 0.3561f, 0.3636f, 0.3711f, 0.3786f, 0.2318f,

                                0.0376f, 0.087f, 0.0894f, 0.0918f, 0.0942f, 0.0883f,
                                0.1476f, 0.2461f, 0.2596f, 0.2731f, 0.2866f, 0.22479999f,
                                0.24459998f, 0.3811f, 0.3946f, 0.40809998f, 0.4216f, 0.3208f,
                                0.34159997f, 0.5161f, 0.5296f, 0.5431f, 0.5566f, 0.4168f,
                                0.4386f, 0.6511001f, 0.6646f, 0.6781f, 0.69159997f, 0.5128f,
                                0.3216f, 0.3561f, 0.3636f, 0.3711f, 0.3786f, 0.2318f,               
                           });


        assertArrayEquals(expected.getValues(), actual.getValues(), 1e-4f); // 0.8432 vs 0.8431999
    }                 
    
    @Test
    public void testBackwardFromTwoConvolutionalChannelsToTwoPoolingChannels() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(12, 12, 1);
        Tensor input = new Tensor(12, 12,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f, 0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f, -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f, -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f, 0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f, 0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f, 0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f, -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f, -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f, 0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f, 0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f, 0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f                             
                });
        
        Tensor filter = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f
                            });
        
        Tensor filter2 = new Tensor(3, 3, 2, 
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f,
                              
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f                              
                            });           

        float[] biases = new float[]{0.0f, 0.0f};

        ConvolutionalLayer prevLayer = new ConvolutionalLayer(3, 3, 2);
        prevLayer.setPrevLayer(inputLayer);        
        prevLayer.activationType = ActivationType.LINEAR;
             
        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(prevLayer);
        prevLayer.setNextlayer(instance);
               
        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 2);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
       
        prevLayer.init();
        instance.init();  
        nextLayer.init();        
        
        prevLayer.filters[0] = filter;
        prevLayer.filters[1] = filter;
        prevLayer.biases = new float[] {0.0f, 0.0f};        
        
        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.filters[0] = filter2;
        nextLayer.filters[1] = filter2;
        nextLayer.biases = new float[] {0.0f, 0.0f};     
        
                           
        // propagate forward and backward
        inputLayer.setInput(input);
        instance.forward();
        nextLayer.forward();
        nextLayer.setDeltas(new Tensor(6, 6, 2,
                                        new float[] { 
                                            0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f,
                                            0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f,
                                            0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
                                            0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f,
                                            0.51f, 0.52f, 0.53f, 0.54f, 0.55f, 0.56f,
                                            0.61f, 0.62f, 0.63f, 0.64f, 0.65f, 0.66f,
                                            
                                            0.22f, 0.24f, 0.26f, 0.28f, 0.3f,  0.32f,
                                            0.42f, 0.44f, 0.46f, 0.48f, 0.5f,  0.52f,
                                            0.62f, 0.64f, 0.66f, 0.68f, 0.7f,  0.72f,
                                            0.82f, 0.84f, 0.86f, 0.88f, 0.9f,  0.92f,
                                            1.02f, 1.04f, 1.06f, 1.08f, 1.1f,  1.12f,
                                            1.22f, 1.24f, 1.26f, 1.28f, 1.3f,  1.32f                                           
                                        }));
        instance.backward();
        
        Tensor actual = instance.getDeltas();

        Tensor expected = new Tensor(6, 6, 2,
                new float[]{
                                0.11280f,  0.26100f,  0.26820f,  0.27540f,  0.28260f,  0.26490f,
                                0.44280f,  0.73830f,  0.77880f,  0.81930f,  0.85980f,  0.67440f,
                                0.73380f,  1.14330f,  1.18380f,  1.22430f,  1.26480f,  0.96240f,
                                1.02480f,  1.54830f,  1.58880f,  1.62930f,  1.66980f,  1.25040f,
                                1.31580f,  1.95330f,  1.99380f,  2.03430f,  2.07480f,  1.53840f,
                                0.96480f,  1.06830f,  1.09080f,  1.11330f,  1.13580f,  0.69540f,
                                
                                0.11280f,  0.26100f,  0.26820f,  0.27540f,  0.28260f,  0.26490f,
                                0.44280f,  0.73830f,  0.77880f,  0.81930f,  0.85980f,  0.67440f,
                                0.73380f,  1.14330f,  1.18380f,  1.22430f,  1.26480f,  0.96240f,
                                1.02480f,  1.54830f,  1.58880f,  1.62930f,  1.66980f,  1.25040f,
                                1.31580f,  1.95330f,  1.99380f,  2.03430f,  2.07480f,  1.53840f,
                                0.96480f,  1.06830f,  1.09080f,  1.11330f,  1.13580f,  0.69540f                              
                           });

        assertArrayEquals(expected.getValues(), actual.getValues(), 1e-4f);
    }        
    
    
}
