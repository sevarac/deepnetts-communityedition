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
     * Doublechecked 19.01.19
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

        /* Conv Layer Output:
Novi
            [ -0.3824276, -0.2446367, 0.11291636, 0.07267165, 0.23936461, 0.36409616,
               0.19804797, 0.42304912, 0.49336636, 0.47916514, 0.45872873, 0.44158658,
               0.20543453, 0.38311034, 0.37342745, 0.1391905, -0.08965858, -0.06610347,
               0.26743573, 0.421899, 0.61740464, 0.75935936, 0.6511543, 0.48039678,
               0.20160024, 0.4124003, 0.2891147, 0.1743989, 0.073865205, 0.22991526,
              -0.028991872, 0.10184565, 0.21136525, 0.044171233, 0.045269042, 0.0064999186

        */

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        Tensor actualOutputs = instance.getOutputs();
        // result of maxpooling applied to prev conv layer output
        Tensor expectedOutputs = new Tensor(3, 3,
                new float[]{
                                0.42304912f, 0.49336636f, 0.45872873f,
                                0.421899f, 0.75935936f, 0.6511543f,
                                0.4124003f, 0.2891147f, 0.22991526f
                });

        /* maxIdxs  1,1     1,2     1,4
                    3,1     3,3     3,4
                    4,1     4,2     4,5  */

        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-8f);
    }

    /**
     * Doublechecked 19.01.19
     */
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
        convLayer.setActivationType(ActivationType.LINEAR);
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
        -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002, 0.0064999983,

        -0.20889999, -0.26760003, -0.010199998, -6.999895E-4,  0.22350001,    0.22450002,
        0.3319,       0.48950002,  0.44680002,   0.4791,       0.40600002,    0.25570002,
        0.19569999,   0.3932,      0.2622,       0.014099985, -0.060699996,  -0.15130001,
        0.2328,       0.3976,      0.6252,       0.6627,       0.8222,        0.4177,
        0.27,         0.31350002,  0.23630002,  -0.0035999827, 0.04750003,    0.10620001,
        0.028599992, 0.105699986,  0.18150005,   0.033699997,  0.064200014,  -0.014600009
         */

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(convLayer);
        instance.init();
        instance.forward();

        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(3, 3, 2,
                new float[]{
                    0.45139998f, 0.5405f,     0.4957f,
                    0.45f,       0.99470013f, 0.77730006f,
                    0.4385f,     0.29759997f, 0.23410001f,

                    0.48950002f, 0.4791f,     0.40600002f,
                    0.3976f,     0.6627f,     0.8222f,
                    0.31350002f, 0.23630002f, 0.10620001f
                });

        /* maxIdxs          1,1     1,2     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5

                            1,1     1,3     1,4
                            3,1     3,3     3,4
                            4,1     4,2     4,5
         */
        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-8f);
    }

    /**
     * Doublechecked with octave 31.1.19.
     */
    @Test
    public void testBackwardFromFullyConnectedToSingleChannel() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);    //
        Tensor input = new Tensor(6, 6,
                new float[]{0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                            0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                            0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                            0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f});

        Tensor filter = new Tensor(3, 3,
                new float[]{ 0.1f, 0.2f, 0.3f,
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
        FullyConnectedLayer nextLayer = new FullyConnectedLayer(2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.setNextlayer(new OutputLayer(1)); // just dummy next layer to prevent exception
        nextLayer.init(); // init weights: [0.18075174, 0.5545214, 0.072818756; 0.31912476, -0.49894053, -0.6323205; 0.2685551, 0.4376064, -0.34319848; 0.11627263, -0.3802099, 0.60284144; 0.15107232, -0.5185875, -0.41857186; 0.7019462, -0.0617525, -0.64165723]
        nextLayer.setDeltas(new Tensor(0.1f, 0.2f));
        // poslednja dimenzija matrice tezina je 2 - koliko ima neurona u fc. - zasto je 3x3x1  X  2  (prev layer x fcCols)
        /*
        weights sa dva neurona u fc:
            0.18075174, 0.5545214, 0.072818756,
            0.31912476, -0.49894053, -0.6323205,
            0.2685551, 0.4376064, -0.34319848,

            0.11627263, -0.3802099, 0.60284144,
            0.15107232, -0.5185875, -0.41857186,
            0.7019462, -0.0617525, -0.64165723
         */

        instance.backward();
        Tensor actualDeltas = instance.getDeltas();   // delte koje su propagirane sa fac lajera unazad na maxpooling, po principu svaki na sve

        // sum (delta * weight) and transpose
        // Test: 0.1 * 0.18075174 + 0.2 * 11627263 = 0.0413297 ...
        Tensor expectedDeltas = new Tensor(3, 3,
                new float[]{ 0.0413297f,   0.062126940f,   0.167244750f,
                            -0.02058984f,  -0.153611553f,   0.031410140f,
                             0.127850164f,  -0.146946422f,  -0.162651294f  });

        assertArrayEquals(expectedDeltas.getValues(), actualDeltas.getValues(),  1e-7f);
    }

    /**
     * Doublechecked with octave 31.1.19.
     */
    @Test
    public void testBackwardFromFullyConnectedToMultiChannels() {
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
        FullyConnectedLayer nextLayer = new FullyConnectedLayer(2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.setNextlayer(new OutputLayer(1)); // just dummy output layer to prevent exception
        nextLayer.init(); // init weights: [0.0862301, -0.28197122, 0.44707918; 0.112038255, -0.38459483, -0.31042123; 0.5205773, -0.04579687, -0.47586578; -0.4501995, -0.4715696, -0.4138012; -0.44830847, -0.1773395, -0.08274263; 0.47323978, 0.41012484, 0.19486493; 0.08227211, -0.54566085, -0.11338204; -1.4930964E-4, -0.26475245, 0.52672565; -0.38034722, 0.3306117, -0.26243588; 0.31195676, -0.04197538, -0.5319252; -0.06671178, -0.29084632, -0.31613958; -0.025327086, 0.12511307, -0.3920598]
        nextLayer.setDeltas(new Tensor(0.1f, 0.2f));
        // poslednja dimenzija matrice tezina je 2 - koliko ima neurona u fc. - zasto je 3x3x1  X  2  (prev layer x fcCols)
        /*
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

        // sum (delta * weight) and transpose
        // Test: 0.1 * 0.0862301 + 0.2 * 0.08227211 =  0.025077432   +
        // Test: 0.1 * -0.28197122 + 0.2 * -0.54566085 = −0.137329292 +
        // Test: 0.1 * -0.4501995 + 0.2 * 0.31195676 = 0.017371402
        // Test: 0.1 * 0.19486493 + 0.2 * -0.3920598 = −0.05892546


        Tensor expected = new Tensor(3, 3, 2,
                new float[]{ 0.025077432f,   0.011173964f,  -0.024011714f,
                            -0.137329292f,  -0.091409973f,   0.061542653f,
                             0.022031510f,   0.074303007f,  -0.100073754f,

                             0.017371402f,  -0.058173203f,   0.042258561f,
                            -0.055552036f,  -0.075903214f,   0.066035098f,
                            -0.147765160f,  -0.071502179f,  -0.058925467f
                });


        assertArrayEquals(actual.getValues(), expected.getValues(), 1e-8f);
    }

    /**
     * Doublechecked with octave 2.2.19.
     * Testira propagaciju delta sa jednog konvolucionog kanala na jedan maxpooling kanal 
     * Proverene pozicije [0,0], [0, 5], [5, 0], [1,1] i [5,5]
     */
    @Test
    public void testBackwardFromSingleConvolutionalToSinglePoolingChannel() {
        RandomGenerator.getDefault().initSeed(123);
        
        // we need input in order to simulate forward propagation
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

        float[] biases = new float[]{0.0f};


        ConvolutionalLayer prevLayer = new ConvolutionalLayer(3, 3, 1);
        prevLayer.setPrevLayer(inputLayer);
        prevLayer.activationType = ActivationType.LINEAR;

        // instance of maxpooling layer to test
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

        Tensor actualDeltas = instance.getDeltas();

        // mislim da ovo nije dobro tj. da ne akumulira sve sto treba
        Tensor expectedDeltas = new Tensor(6, 6,
                new float[]{
                                0.0376f, 0.087f, 0.0894f, 0.0918f, 0.0942f, 0.0883f,
                                0.1476f, 0.2461f, 0.2596f, 0.2731f, 0.2866f, 0.22479999f,
                                0.24459998f, 0.3811f, 0.3946f, 0.40809998f, 0.4216f, 0.3208f,
                                0.34159997f, 0.5161f, 0.5296f, 0.5431f, 0.5566f, 0.4168f,
                                0.4386f, 0.6511001f, 0.6646f, 0.6781f, 0.69159997f, 0.5128f,
                                0.3216f, 0.3561f, 0.3636f, 0.3711f, 0.3786f, 0.2318f
                           });

        assertArrayEquals(expectedDeltas.getValues(), actualDeltas.getValues(), 1e-7f);
    }

    /**
     * Doublechecked with octave 4.3.2019.  
     */
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
        
        Tensor[] filter2 = new Tensor[2];
        
        filter2[0] = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f
                            });

        filter2[1] = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.19f, -0.13f,
                              0.4f,   0.5f,  0.21f
                            });
        

        ConvolutionalLayer prevLayer = new ConvolutionalLayer(3, 3, 1);
        prevLayer.setPrevLayer(inputLayer);
        prevLayer.activationType = ActivationType.LINEAR;

        MaxPoolingLayer instance = new MaxPoolingLayer(2, 2, 2);
        instance.setPrevLayer(prevLayer);
        prevLayer.setNextlayer(instance);

        // 2 channel conv layer as next layer
        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 2); 
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);

        prevLayer.init();
        instance.init();
        nextLayer.init();

        prevLayer.filters[0] = filter;
        prevLayer.biases = new float[]{0.0f};

        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.filters[0] = filter2[0];
        nextLayer.filters[1] = filter2[1];
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
                            0.0974f,   0.2442f,   0.25f,   0.2558f,   0.2616f,   0.2425f,
                            0.4134f,   0.7075f,   0.7466f, 0.7857f,   0.8248f,   0.638f,
                            0.6904f,   1.0985f,   1.1376f, 1.1767f,   1.2158f,   0.912f,
                            0.9674f,   1.4895f,   1.5286f, 1.5677f,   1.6068f,   1.186f,
                            1.2444f,   1.8805f,   1.9196f, 1.9587f,   1.9978f,   1.46f,
                            0.8794f,   0.9815f,   1.0026f, 1.0237f,   1.0448f,   0.603f  
                           });
        
        assertArrayEquals(expected.getValues(), actual.getValues(), 1e-4f);
    }

    /**
     * Test delta propagation from single convolutional channel back to two max pooling channels.
     * Double checked with octave 4.2.19.
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
                             -0.11f, -0.19f, -0.13f,
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

                                0.0299f, 0.078600004f, 0.0803f, 0.08200001f, 0.0837f, 0.0771f,
                                0.1329f, 0.2307f, 0.2435f, 0.2563f, 0.2691f, 0.2066f,
                                0.22289999f, 0.3587f, 0.3715f, 0.3843f, 0.3971f, 0.2956f,
                                0.31289998f, 0.4867f, 0.4995f, 0.51229995f, 0.5251f, 0.3846f,
                                0.4029f, 0.6147f, 0.6275f, 0.6403f, 0.6531f, 0.4736f,
                                0.2789f, 0.3127f, 0.31949997f, 0.3263f, 0.3331f, 0.1856f
                           });

        assertArrayEquals(expected.getValues(), actual.getValues(), 1e-7f);
    }

    /**
     * Doublechecked with octave 3.4.2019.  
     */
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

        // ovde treba da imamo 2 filtera
        Tensor[] filter2 = new Tensor[2];
        
        filter2[0] = new Tensor(3, 3, 2, 
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f,
                              
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.13f, -0.13f,
                              0.4f,   0.5f,  0.21f                              
                            });

        filter2[1] = new Tensor(3, 3, 2,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f,
                              
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.13f, -0.13f,
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
        nextLayer.filters[0] = filter2[0];
        nextLayer.filters[1] = filter2[1];
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
                                0.0752f,   0.174f,   0.1788f,   0.1836f,   0.1884f,   0.1766f,
                                0.2952f,   0.4922f,   0.5192f,   0.5462f,   0.5732f,   0.4496f,
                                0.4892f,   0.7622f,   0.7892f,   0.8162f,   0.8432f,   0.6416f,
                                0.6832f,   1.0322f,   1.0592f,   1.0862f,   1.1132f,   0.8336f,
                                0.8772f,   1.3022f,   1.3292f,   1.3562f,   1.3832f,   1.0256f,
                                0.6432f,   0.7122f,   0.7272f,   0.7422f,   0.7572f,   0.4636f,

                                0.073f,   0.1716f,   0.1762f,   0.1808f,   0.1854f,   0.1734f,
                                0.291f,   0.4878f,   0.5146f,   0.5414f,   0.5682f,   0.4444f,
                                0.483f,   0.7558f,   0.7826f,   0.8094f,   0.8362f,   0.6344f,
                                0.675f,   1.0238f,   1.0506f,   1.0774f,   1.1042f,   0.8244f,
                                0.867f,   1.2918f,   1.3186f,   1.3454f,   1.3722f,   1.0144f,
                                0.631f,   0.6998f,   0.7146f,   0.7294f,   0.7442f,   0.4504f
                           });
        
        assertArrayEquals(expected.getValues(), actual.getValues(), 1e-4f);
    }


}
