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
public class ConvolutionalLayerTest {
      
    /**
     * 2D convolution as weighted sum of inputs and filter weights, at each filter position.
     * Sliding filter over entire input matrix.
     * 
     * [0, 0]:  -0.2 * 0.3 + -0.3 * 0.5 + 0.5 * 0.1 + -0.6 * 0.21 = -0.286  
     * [1, 1]:  0.3 * 0.1 + 0.5 * 0.2 + 0.6 * 0.3 + 0.1 * -0.11 + -0.6 * -0.2 + 0.51 * -0.3 + 0.28 * 0.4 + 0.61 * 0.5  + -0.15 * 0.21 = 0.6515
     * [0, 4]: 0.14 * -0.2 + 0.14 *0.5 + 0.2 * -0.11 + 0.23 * 0.4 = 0.112
     * [4, 0]: 0.62 * -0.2 + 0.46 * 0.2 + 0.72 * 0.3  + 0.31 * -0.3 = 0.091
     * [4, 4]: 0.53 * -0.2 + 0.17 * -0.11 + 0.1 * 0.43 + 0.34 * 0.2  = −0.0137
     * [2, 2]: -0.6 * 0.1 + 0.51 *0.2 + 0.23 * 0.3 + -0.11 * 0.61 + -0.2 * -0.15 + -0.3 * 0.47 + 0.4 *0.72 + 0.61 * 0.5  + 0.43 * 0.21 = 0.6162
     * 
     */
    @Test
    public void testForwardSingleOutputChannelSingleFilter() {
        InputLayer inputLayer = new InputLayer(5, 5, 1);
        Tensor input = new Tensor(5, 5, 
                                new float[] {0.3f,  0.5f,   0.6f,  0.2f,  0.14f,
                                             0.1f, -0.6f,   0.51f, 0.23f, 0.14f,
                                             0.28f, 0.61f, -0.15f, 0.47f, 0.34f,
                                             0.46f, 0.72f,  0.61f, 0.43f, 0.34f, 
                                             0.62f, 0.31f, -0.25f, 0.17f, 0.53f});
        
        float[] biases = new float[] {0.0f};
        
        Tensor filter = new Tensor(3, 3,
                                new float[] {0.1f,   0.2f,  0.3f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.4f,  0.5f,   0.21f});        

        Tensor expectedOutputs = new Tensor(5, 5,
                                new float[] {-0.286f, -0.4659f, -0.1717f, 0.2004f, 0.112f,
                                              0.6381f, 0.6515f, 0.3927f, 0.2443f, 0.3527f,
                                             -0.0178f, 0.6073f, 0.6162f, 0.4899f, 0.2733f,
                                              0.3061f, 0.0779f, -0.1235f, 0.0222f, 0.3327f,
                                              0.091f, 0.3178f, 0.2879f, 0.0835f, -0.0137f});
                
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.setActivationType(ActivationType.LINEAR);
        instance.init();
        instance.filters[0] = filter;
        instance.biases = biases;
                
        inputLayer.setInput(input);
        instance.forward();
        Tensor actualOutputs = instance.getOutputs();
        
        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-7f);               
    }
    
    // same as above but with different values - used for testint other cases
   @Ignore
    public void testForwardSingleOutputChannelSingleFilter2() {
        InputLayer inputLayer = new InputLayer(5, 5, 1);
//        Tensor input = new Tensor(5, 5, 
//                                new float[] {0.1f,  0.8f,   0.3f,  0.3f,  0.24f,
//                                             0.2f, -0.4f,   0.41f, 0.33f, 0.34f,
//                                             0.33f, 0.71f, -0.25f, 0.57f, 0.14f,
//                                             0.56f, 0.32f,  0.71f, 0.73f, 0.74f, 
//                                             0.72f, 0.11f, -0.35f, 0.27f, 0.23f});

        Tensor input = new Tensor(5, 5, 
                                new float[] {0.3f,  0.5f,   0.6f,  0.2f,  0.14f,
                                             0.1f, -0.6f,   0.51f, 0.23f, 0.14f,
                                             0.28f, 0.61f, -0.15f, 0.47f, 0.34f,
                                             0.46f, 0.72f,  0.61f, 0.43f, 0.34f, 
                                             0.62f, 0.31f, -0.25f, 0.17f, 0.53f});
        
        float[] biases = new float[] {0.0f};
        
        Tensor filter = new Tensor(3, 3,
                                new float[] {0.2f,   0.1f,  0.4f,
                                            -0.12f, -0.1f, -0.5f,
                                             0.2f,  0.3f,   0.71f});        

//        Tensor expectedOutputs = new Tensor(5, 5,
//                                new float[] {-0.634f, -0.0309f, 0.0012999f, 0.2364f, 0.108f,
//                                              1.1130999f, 0.1325f, 0.62369996f, 0.1542f, 0.1664f,
//                                              -0.1328f, 0.8905f, 0.5431f, 1.0404f, 0.3856f, 
//                                              0.3951f, -0.48869997f, -0.020700023f, -0.2909f, 0.0894f,
//                                              0.057f, 0.5056f, 0.3138f, 0.411f, 0.1646f});        
        
        Tensor expectedOutputs = new Tensor(5, 5,
                                new float[] {-0.676f, -0.1839f, -0.0237f, 0.1084f, 0.05f,
                                              1.0371f, 0.2755f, 0.5567f, 0.3942f, 0.2084f,
                                              0.0862f, 0.8855f, 0.3621f, 0.47439998f, 0.1576f,
                                              0.2721f, -0.3357f, 0.0403f, 0.24409999f, 0.2354f,
                                              0.117f, 0.4276f, 0.2798f, 0.049f, 0.0466f});
                
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.setActivationType(ActivationType.LINEAR);
        instance.init();
        instance.filters[0] = filter;
        instance.biases = biases;
                
        inputLayer.setInput(input);
        instance.forward();
        Tensor actualOutputs = instance.getOutputs();
        
        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-7f);               
    }    
   
    @Test
    public void testForwardSingleOutputChannelTwoInputChannels() {
        InputLayer inputLayer = new InputLayer(5, 5, 2);
        Tensor input = new Tensor(5, 5, 2,
                                new float[] {0.3f,  0.5f,   0.6f,  0.2f,  0.14f,
                                             0.1f, -0.6f,   0.51f, 0.23f, 0.14f,
                                             0.28f, 0.61f, -0.15f, 0.47f, 0.34f,
                                             0.46f, 0.72f,  0.61f, 0.43f, 0.34f, 
                                             0.62f, 0.31f, -0.25f, 0.17f, 0.53f,
                                
                                             0.1f,  0.8f,   0.3f,  0.3f,  0.24f,
                                             0.2f, -0.4f,   0.41f, 0.33f, 0.34f,
                                             0.33f, 0.71f, -0.25f, 0.57f, 0.14f,
                                             0.56f, 0.32f,  0.71f, 0.73f, 0.74f, 
                                             0.72f, 0.11f, -0.35f, 0.27f, 0.23f                                                                                                
                                });

        float[] biases = new float[] {0.0f, 0.0f};
        
        Tensor filter1 = new Tensor(3, 3, 2,
                                new float[] {0.1f,   0.2f,  0.3f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.4f,  0.5f,   0.21f,
                                
                                             0.2f,   0.1f,  0.4f,
                                            -0.12f, -0.1f, -0.5f,
                                             0.2f,  0.3f,   0.71f
                                });        
                           
        // use linear tf
        Tensor expectedOutputs = new Tensor(5, 5,
                                new float[] {-0.92f,   -0.49680f, -0.17040f,  0.43680f,  0.22f,
                                              1.7512f,  0.78400f,  1.01640f,  0.39850f,  0.51910f,
                                             -0.1506f,  1.49780f,  1.15930f,  1.53030f,  0.65890f,
                                              0.7012f, -0.41080f, -0.14420f, -0.26870f,  0.42210f,
                                              0.148f,   0.82340f,  0.60170f,  0.49450f,  0.15090f });
                
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.setActivationType(ActivationType.LINEAR);
        instance.init();
        instance.filters[0] = filter1;
        instance.biases = biases;
                
        inputLayer.setInput(input);
        instance.forward();
        Tensor actualOutputs = instance.getOutputs();
        
        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-6f);               
    }    
    
    // isti kao ovaj iznad samo sa drugim vrednostima za potrebe daljeg testiranja
    @Ignore
    public void testForwardSingleOutputChannelTwoInputChannels2() {
        InputLayer inputLayer = new InputLayer(5, 5, 2);
        Tensor input = new Tensor(5, 5, 2,
                                new float[] {0.3f,  0.5f,   0.6f,  0.2f,  0.14f,
                                             0.1f, -0.6f,   0.51f, 0.23f, 0.14f,
                                             0.28f, 0.61f, -0.15f, 0.47f, 0.34f,
                                             0.46f, 0.72f,  0.61f, 0.43f, 0.34f, 
                                             0.62f, 0.31f, -0.25f, 0.17f, 0.53f,
                                
                                             0.1f,  0.8f,   0.3f,  0.3f,  0.24f,
                                             0.2f, -0.4f,   0.41f, 0.33f, 0.34f,
                                             0.33f, 0.71f, -0.25f, 0.57f, 0.14f,
                                             0.56f, 0.32f,  0.71f, 0.73f, 0.74f, 
                                             0.72f, 0.11f, -0.35f, 0.27f, 0.23f                                                                                                
                                });

        float[] biases = new float[] {0.0f, 0.0f};
        
        Tensor filter1 = new Tensor(3, 3, 2,
                                new float[] {0.2f,   0.6f,  0.5f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.42f,  0.13f, 0.21f,
                                
                                             0.27f,   0.15f,  0.41f,
                                            -0.22f, -0.1f, -0.27f,
                                             0.21f,  0.13f,   0.71f
                                });        
                           
        // use linear tf
        Tensor expectedOutputs = new Tensor(5, 5,
                                new float[] {-0.807f, -0.14380005f, -0.4558f, 0.3351f, 0.0883f,
                                              1.7325f, 0.9208f, 1.6535997f, 0.3658f, 0.4606f,
                                              -0.3267001f, 0.9713999f, 1.0763999f, 1.5432f, 0.4853f,
                                              0.6805999f, -0.11919996f, 0.28889996f, -0.0334f, 0.34989995f,
                                              0.5325f, 1.1891999f, 1.12f, 0.977f, 0.391f });
                
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.setActivationType(ActivationType.LINEAR);
        instance.init();
        instance.filters[0] = filter1;
        instance.biases = biases;
                
        inputLayer.setInput(input);
        instance.forward();
        Tensor actualOutputs = instance.getOutputs();
        
        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-6f);               
    }        
  
    @Test
    public void testForwardTwoOutputChannelsSingleInputChannel() {
        InputLayer inputLayer = new InputLayer(5, 5, 1);
        Tensor input = new Tensor(5, 5, 
                                new float[] {0.3f,  0.5f,   0.6f,  0.2f,  0.14f,
                                             0.1f, -0.6f,   0.51f, 0.23f, 0.14f,
                                             0.28f, 0.61f, -0.15f, 0.47f, 0.34f,
                                             0.46f, 0.72f,  0.61f, 0.43f, 0.34f, 
                                             0.62f, 0.31f, -0.25f, 0.17f, 0.53f});

        float[] biases = new float[] {0.0f, 0.0f};
        
        Tensor filter1 = new Tensor(3, 3,
                                new float[] {0.1f,   0.2f,  0.3f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.4f,  0.5f,   0.21f});        
        Tensor filter2 = new Tensor(3, 3,
                                new float[] { 0.2f,   0.1f,  0.4f,
                                             -0.12f, -0.1f, -0.5f,
                                              0.2f,  0.3f,   0.71f });                
        
        Tensor expectedOutputs = new Tensor(5, 5, 2,
                                new float[] {-0.286f, -0.4659f, -0.1717f, 0.2004f, 0.112f,
                                              0.6381f, 0.6515f, 0.3927f, 0.2443f, 0.3527f,
                                             -0.0178f, 0.6073f, 0.6162f, 0.4899f, 0.2733f,
                                              0.3061f, 0.0779f, -0.1235f, 0.0222f, 0.3327f,
                                              0.091f, 0.3178f, 0.2879f, 0.0835f, -0.0137f,
                                
                                             -0.676f, -0.1839f, -0.0237f, 0.1084f, 0.05f,
                                              1.0371f, 0.2755f, 0.5567f, 0.3942f, 0.2084f,
                                              0.0862f, 0.8855f, 0.3621f, 0.47439998f, 0.1576f,
                                              0.2721f, -0.3357f, 0.0403f, 0.24409999f, 0.2354f,
                                              0.117f, 0.4276f, 0.2798f, 0.049f, 0.0466f                                                                                                
                                });
                
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 2);
        instance.setPrevLayer(inputLayer);
        instance.setActivationType(ActivationType.LINEAR);
        instance.init();
        instance.filters[0] = filter1;
        instance.filters[1] = filter2;
        instance.biases = biases;
                
        inputLayer.setInput(input);
        instance.forward();
        Tensor actualOutputs = instance.getOutputs();
        
        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-7f);               
    }        
    
   @Test
    public void testForwardTwoOutputChannelsTwoInputChannels() {
        InputLayer inputLayer = new InputLayer(5, 5, 2);
        Tensor input = new Tensor(5, 5, 2,
                                new float[] {0.3f,  0.5f,   0.6f,  0.2f,  0.14f,
                                             0.1f, -0.6f,   0.51f, 0.23f, 0.14f,
                                             0.28f, 0.61f, -0.15f, 0.47f, 0.34f,
                                             0.46f, 0.72f,  0.61f, 0.43f, 0.34f, 
                                             0.62f, 0.31f, -0.25f, 0.17f, 0.53f,
                                
                                             0.1f,  0.8f,   0.3f,  0.3f,  0.24f,
                                             0.2f, -0.4f,   0.41f, 0.33f, 0.34f,
                                             0.33f, 0.71f, -0.25f, 0.57f, 0.14f,
                                             0.56f, 0.32f,  0.71f, 0.73f, 0.74f, 
                                             0.72f, 0.11f, -0.35f, 0.27f, 0.23f                                                                                               
                                });

        // set biases to zero
        float[] biases = new float[] {0.0f, 0.0f};
        
        Tensor filter1 = new Tensor(3, 3, 2,
                                new float[] {0.1f,   0.2f,  0.3f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.4f,  0.5f,   0.21f,
                                
                                             0.2f,   0.1f,  0.4f,
                                            -0.12f, -0.1f, -0.5f,
                                             0.2f,  0.3f,   0.71f
                                });    
        
        
        Tensor filter2 = new Tensor(3, 3, 2,
                                new float[] {0.2f,   0.6f,  0.5f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.42f,  0.13f, 0.21f,
                                
                                             0.27f,   0.15f,  0.41f,
                                            -0.22f, -0.1f, -0.27f,
                                             0.21f,  0.13f,   0.71f
                                });             
                           
        // use linear tf
        Tensor expected = new Tensor(5, 5, 2,
                                new float[] {-0.92f,   -0.49680f, -0.17040f,  0.43680f,  0.22f,
                                              1.7512f,  0.78400f,  1.01640f,  0.39850f,  0.51910f,
                                             -0.1506f,  1.49780f,  1.15930f,  1.53030f,  0.65890f,
                                              0.7012f, -0.41080f, -0.14420f, -0.26870f,  0.42210f,
                                              0.148f,   0.82340f,  0.60170f,  0.49450f,  0.15090f,
                                
                                             -0.807f, -0.14380005f, -0.4558f, 0.3351f, 0.0883f,
                                              1.7325f, 0.9208f, 1.6535997f, 0.3658f, 0.4606f,
                                              -0.3267001f, 0.9713999f, 1.0763999f, 1.5432f, 0.4853f,
                                              0.6805999f, -0.11919996f, 0.28889996f, -0.0334f, 0.34989995f,
                                              0.5325f, 1.1891999f, 1.12f, 0.977f, 0.391f                               
                                });
                        
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 2);
        instance.setPrevLayer(inputLayer);
        instance.setActivationType(ActivationType.LINEAR);
        instance.init();
        instance.filters[0] = filter1;
        instance.filters[1] = filter2;
        instance.biases = biases;
                
        inputLayer.setInput(input);
        instance.forward();
        Tensor actual = instance.getOutputs();
        
        assertArrayEquals(expected.getValues(), actual.getValues(),1e-6f);             
    }   
    
    
    @Test
    public void testBackwardFromFullyConnectedToSingleChannel() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f
                });

        Tensor filter = new Tensor(3, 3,
                new float[]{  0.1f, 0.2f, 0.3f,
                             -0.11f, -0.2f, -0.3f,
                              0.4f, 0.5f, 0.21f
                            });

        float[] biases = new float[]{0.0f};

        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.activationType = ActivationType.LINEAR;
        instance.init();
        instance.filters[0] = filter;
        instance.biases = biases;
      
        DenseLayer nextLayer = new DenseLayer(2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.init(); // init weights               
        
        // poslednja dimenzija matrice tezina je 2 - koliko ima neurona u fc. - zasto je 3x3x1  X  2  (prev layer x fcCols)
        /* test samo sa jednim neuronom u fc i delta 0.1, pomnozi sve tezine sa 0.1
        
        weights sa dva neurona u fc:
            0.09724945, 0.2983478, 0.03917849, 0.17169794, -0.26844376, -0.34020588,
            0.1444901, 0.23544434, -0.18465026, 0.062557876, -0.20456341, 0.32434532,
            0.081281066, -0.2790144, -0.2252032, 0.37766644, -0.033224553, -0.3452293,
           -0.32660902, -0.3421125, -0.3002029, -0.32523713, -0.12865558, -0.060027808,
            0.3433242, 0.2975358, 0.14136991, 0.059686452, -0.39586398, -0.08225599,
           -1.0833144E-4, -0.1920716, 0.38212696, -0.27593285, 0.23985091, -0.19039099,
            
            0.2263172, -0.030452132, -0.38589907, -0.04839781, -0.21100208, -0.22935173,
            -0.018374175, 0.09076658, -0.28443003, -0.37077007, -0.04809025, 0.1501194,
            -0.30833668, 0.14631799, 0.27727768, -0.33754998, -0.27284825, 0.3550851,
            -0.0791384, -0.050165385, 0.3491011, 0.15402898, -0.066467196, 0.1401873,
             0.26111117, 0.10234681, 0.1295233, 0.05936882, -0.37287155, -0.0890311,
             0.21884283, -0.24705583, -0.06623617, -0.37612358, -0.25002092, -0.37228096
        
         */

        inputLayer.setInput(input);
        instance.forward();
        nextLayer.setDeltas(new Tensor(0.1f, 0.2f)); // test with  (0.1f, 0.0f) (0.0f, 0.1f) (0.1f, 0.2f)  - transponuje tezine u odnosu na ovo gore
        instance.backward();
        
        Tensor actual = instance.getDeltas();

        // sum delta * weight and transpose
        // Test: 0.1 * 0.18075174 + 0.2 * 11627263 = 0.0413297 ... 
        Tensor expected = new Tensor(6, 6,
                new float[]{
                            0.054988384f, 0.010774175f, -0.053539228f, -0.04848858f, 0.086554654f, 0.043757733f,
                            0.023744354f, 0.041697748f, 0.001362158f, -0.04424433f, 0.05022294f, -0.06861833f,
                           -0.07326196f, -0.07535103f, 0.032935217f, 0.03979993f, 0.04004165f, 0.024965465f,
                            0.0074902317f, -0.06789822f, -0.029743355f, -0.0017179176f, 0.017842408f, -0.102818005f,
                           -0.06904479f, -0.030074392f, -0.057892106f, -0.026158998f, -0.11416072f, -0.026019093f,
                           -0.07989094f, 0.06245841f, 0.036494087f, 0.022034679f, -0.02603182f, -0.093495294f
                           });

        assertArrayEquals(expected.getValues(), actual.getValues(),1e-7f);
    }    
    
    @Test
    public void testBackwardFromFullyConnectedToTwoChannels() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f
                });

        Tensor filter1 = new Tensor(3, 3,
                                new float[] {0.1f,   0.2f,  0.3f,
                                            -0.11f, -0.2f, -0.3f,
                                             0.4f,  0.5f,   0.21f});        
        Tensor filter2 = new Tensor(3, 3,
                                new float[] { 0.2f,   0.1f,  0.4f,
                                             -0.12f, -0.1f, -0.5f,
                                              0.2f,  0.3f,   0.71f });  

        float[] biases = new float[]{0.0f, 0.0f};

        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 2);
        instance.setPrevLayer(inputLayer);
        instance.activationType = ActivationType.LINEAR;
        instance.init();
        instance.filters[0] = filter1;
        instance.filters[1] = filter2;
        instance.biases = biases;
      
        DenseLayer nextLayer = new DenseLayer(2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.init(); // init weights               
        
        /* 4 dimenzije
            weights 6,6,2,2
            0.04482889, -0.14658985, 0.23242536, 0.058245897, -0.1999413, -0.16138029, 0.27063522, -0.023808658, -0.24739082, -0.23404756, -0.24515733, -0.21512498, -0.23306446, -0.09219441, -0.043015838, 0.24602565, 0.21321374, 0.101305455, 0.04277122, -0.28367555, -0.05894451, -7.763505E-5, -0.13763823, 0.27383164, -0.19773309, 0.17187682, -0.13643391, 0.1621786, -0.021821946, -0.2765347, -0.034681797, -0.1512038, -0.16435313, -0.013166904, 0.06504318, -0.20382217, -0.2656933, -0.03446141, 0.10757536, -0.22095363, 0.104851276, 0.19869676, -0.24188784, -0.19552267, 0.2544535, -0.05671045, -0.035948455, 0.25016537, 0.110376924, -0.047630295, 0.100457996, 0.18711188, 0.07334158, 0.092816204, 0.04254359, -0.26719922, -0.06379956, 0.15682247, -0.17703986, -0.047464743, -0.2695296, -0.17916465, -0.266776, 0.06411588, 0.1881656, -0.05855225, 0.13672778, 0.18541706, 0.25141618, 0.18511447, 0.18562087, -0.10097939, 0.0080911815, 0.162958, 0.053575724, 0.07007885, -0.15474628, -0.083920464, -0.11209433, -0.18861544, 0.11107698, 0.13313091, -0.04085371, -0.03972672, 0.26508692, -0.036730155, 0.1283817, -0.14586279, -0.093876794, -0.12281649, 0.26099333, -0.07629384, -0.008357763, -0.11859423, -0.14489461, -0.16762337, 0.11117834, -0.023995668, -0.15978116, 0.25392184, -0.22142889, 0.1641522, 0.03775543, 0.00638026, 0.008377761, 0.13569948, -0.19670889, -0.092926994, 0.029650956, 0.20377985, -0.13487613, 0.09324062, -0.12478796, 0.18674242, -0.048004836, -0.19777527, -0.18909906, -0.14504758, 0.18957362, -0.1267012, -0.11254707, -0.17451146, -0.030494332, -0.26384082, 0.13578412, 0.14597127, -0.2747873, 0.054266065, -0.18481383, -0.008294076, 0.2741039, -0.21448438, 0.25186822, -0.068416506, 0.19974676, -0.16921143, 0.14771613, 0.28284958, -0.24497703, -0.22874713, -0.007346958, 0.21790722, -0.18534103, -0.2403844
        ---
        1. neuron   -- razbi ovo da vidis kako se zapravo raspodeljuju po neuronima, kanalima itd.
            1. channel        
            0.04482889, -0.14658985, 0.23242536, 0.058245897, -0.1999413, -0.16138029,
            0.27063522, -0.023808658, -0.24739082, -0.23404756, -0.24515733, -0.21512498,
           -0.23306446, -0.09219441, -0.043015838, 0.24602565, 0.21321374, 0.101305455,
            0.04277122, -0.28367555, -0.05894451, -7.763505E-5, -0.13763823, 0.27383164,
           -0.19773309, 0.17187682, -0.13643391, 0.1621786, -0.021821946, -0.2765347,
           -0.034681797, -0.1512038, -0.16435313, -0.013166904, 0.06504318, -0.20382217,
            2. channel
           -0.2656933, -0.03446141, 0.10757536, -0.22095363, 0.104851276, 0.19869676,
           -0.24188784, -0.19552267, 0.2544535, -0.05671045, -0.035948455, 0.25016537,
            0.110376924, -0.047630295, 0.100457996, 0.18711188, 0.07334158, 0.092816204,
            0.04254359, -0.26719922, -0.06379956, 0.15682247, -0.17703986, -0.047464743,
           -0.2695296, -0.17916465, -0.266776, 0.06411588, 0.1881656, -0.05855225,
            0.13672778, 0.18541706, 0.25141618, 0.18511447, 0.18562087, -0.10097939,
        
        2. neuron
            1. channel
            0.0080911815, 0.162958, 0.053575724, 0.07007885, -0.15474628, -0.083920464,
           -0.11209433, -0.18861544, 0.11107698, 0.13313091, -0.04085371, -0.03972672,
            0.26508692, -0.036730155, 0.1283817, -0.14586279, -0.093876794, -0.12281649,
            0.26099333, -0.07629384, -0.008357763, -0.11859423, -0.14489461, -0.16762337,
            0.11117834, -0.023995668, -0.15978116, 0.25392184, -0.22142889, 0.1641522,
            0.03775543, 0.00638026, 0.008377761, 0.13569948, -0.19670889, -0.092926994,
            2. channel
            0.029650956, 0.20377985, -0.13487613, 0.09324062, -0.12478796, 0.18674242,
           -0.048004836, -0.19777527, -0.18909906, -0.14504758, 0.18957362, -0.1267012,
           -0.11254707, -0.17451146, -0.030494332, -0.26384082, 0.13578412, 0.14597127,
           -0.2747873, 0.054266065, -0.18481383, -0.008294076, 0.2741039, -0.21448438,
            0.25186822, -0.068416506, 0.19974676, -0.16921143, 0.14771613, 0.28284958,
           -0.24497703, -0.22874713, -0.007346958, 0.21790722, -0.18534103, -0.2403844                        
         */

        inputLayer.setInput(input);
        instance.forward();
        nextLayer.setDeltas(new Tensor(0.1f, 0.1f)); // test with  (0.1f, 0.0f) (0.0f, 0.1f) (0.1f, 0.2f)  - transponuje tezine u odnosu na ovo gore
        instance.backward();
        
        Tensor actual = instance.getDeltas();

        // sum delta * weight and transpose
        Tensor expected = new Tensor(6, 6, 2,
                new float[]{
                            0.00529201f,  0.01585409f,  0.00320225f, 0.03037646f, -0.00865548f, 0.00030736f,
                            0.00163681f, -0.02124241f, -0.01289246f, -0.03599694f,  0.01478812f, -0.01448235f,
                            0.02860011f, -0.01363138f,  0.00853659f, -0.00673023f, -0.02962151f, -0.01559754f,
                            0.01283247f, -0.01009166f,  0.01001629f, -0.01186719f, 0.04161004f,  0.01225326f,
                            -0.03546876f, -0.0286011f,   0.01193369f, -0.02825328f, -0.02432508f, -0.01316657f,
                            -0.02453008f, -0.02548517f, -0.0021511f,   0.01062083f, -0.01123825f, -0.02967492f,
                            
                            -0.023604235f, -0.028989268f, -2.1701492E-4f, -0.02322437f, -0.0017661396f, -0.010824924f,
                            0.016931843f, -0.039329793f, -0.022214176f, -0.021293316f, -0.024758115f, -0.004333006f,
                            -0.002730078f, 0.0065354444f, 0.006996366f, -0.02486134f, -0.0067029223f, 0.024406921f,
                            -0.012771302f, -0.020175803f, -0.0076728947f, 0.01485284f, -0.010509556f, 0.040302172f,
                            -0.0019936683f, 0.015362516f, 0.020912569f, 0.009706406f, 0.033588175f, 2.798438E-5f,
                            0.038543917f, 0.012346416f, 0.023878748f, -0.026194913f, 0.022429733f, -0.034136377f                         
                           });

        assertArrayEquals(expected.getValues(), actual.getValues(),1e-7f);
    }    
        
   @Test
    public void testBackwardFromMaxPoolingToSingleChannel() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f
                });

        Tensor filter = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.2f, -0.3f,
                              0.4f,   0.5f,  0.21f
                            });

        float[] biases = new float[]{0.0f};

        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.activationType = ActivationType.LINEAR;
        instance.init();
        instance.filters[0] = filter;
        instance.biases = biases;
             
        MaxPoolingLayer nextLayer = new MaxPoolingLayer(2, 2, 2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.init(); // init weights               
        
        inputLayer.setInput(input);
        instance.forward();
        nextLayer.forward();
        nextLayer.backward();
        nextLayer.setDeltas(new Tensor( 3,3,
                                        new float[]{ 
                                            0.4f, 0.6f, 0.8f,
                                            0.24f, 0.26f, 0.28f,
                                            0.44f, 0.46f, 0.48f,                                                
                                        })); // test with  (0.1f, 0.0f) (0.0f, 0.1f) (0.1f, 0.2f)  - transponuje tezine u odnosu na ovo gore
        instance.backward();
        
        Tensor actual = instance.getDeltas();

        /* conv layer outputs
           [-0.40289998, -0.24970004, 0.11339998, 0.072799996, 0.2441,      0.38160002,
             0.20070001,  0.45139998, 0.5405,     0.52190006,  0.4957,      0.4742, 
             0.2084,      0.4037,     0.39240003, 0.1401,     -0.08989998, -0.066199996,
             0.27409998,  0.45,       0.72080004, 0.99470013,  0.77730006,  0.52349997,
             0.2044,      0.4385,     0.29759997, 0.1762,      0.074000016, 0.23410001,
            -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002,  0.0064999983]*/        
        
        /* maxIdxs  1,1     1,2     1,4
                    3,1     3,3     3,4
                    4,1     4,2     4,5  */        
        
        // sum delta * weight and transpose
        // Test: 0.1 * 0.18075174 + 0.2 * 11627263 = 0.0413297 ... 
        Tensor expected = new Tensor(6, 6,
                new float[]{
                            0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
                            0.0f, 0.4f,  0.6f,  0.0f,  0.8f,  0.0f,
                            0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
                            0.0f, 0.24f, 0.0f,  0.26f, 0.28f, 0.0f,
                            0.0f, 0.44f, 0.46f, 0.0f,  0.0f,  0.48f,
                            0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f
                           });

        assertArrayEquals(expected.getValues(), actual.getValues(),1e-7f);
    }        
    
    
    @Test
    public void testBackwardFromMaxPoolingToTwoChannels() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f
                });

        Tensor filter = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.2f, -0.3f,
                              0.4f,   0.5f,  0.21f
                            });

        float[] biases = new float[]{0.0f, 0.0f};

        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 2);
        instance.setPrevLayer(inputLayer);
        instance.activationType = ActivationType.LINEAR;
        instance.init();
        instance.filters[0] = filter;
        instance.filters[1] = filter;
        instance.biases = biases;
             
        MaxPoolingLayer nextLayer = new MaxPoolingLayer(2, 2, 2);
        instance.setNextlayer(nextLayer);
        nextLayer.setPrevLayer(instance);
        nextLayer.init();          
        
        inputLayer.setInput(input);
        instance.forward();
        nextLayer.forward();
        nextLayer.backward();
        nextLayer.setDeltas(new Tensor( 3, 3, 2,
                                        new float[]{ 
                                            0.4f, 0.6f, 0.8f,
                                            0.24f, 0.26f, 0.28f,
                                            0.44f, 0.46f, 0.48f,                                                
                                            
                                            0.5f, 0.7f, 0.9f,
                                            0.25f, 0.27f, 0.29f,
                                            0.45f, 0.47f, 0.49f,                                                                                            
                                        }));
        instance.backward();
        
        Tensor actual = instance.getDeltas();

        /* conv layer outputs
           [-0.40289998, -0.24970004, 0.11339998, 0.072799996, 0.2441,      0.38160002,
             0.20070001,  0.45139998, 0.5405,     0.52190006,  0.4957,      0.4742, 
             0.2084,      0.4037,     0.39240003, 0.1401,     -0.08989998, -0.066199996,
             0.27409998,  0.45,       0.72080004, 0.99470013,  0.77730006,  0.52349997,
             0.2044,      0.4385,     0.29759997, 0.1762,      0.074000016, 0.23410001,
            -0.029000014, 0.10220002, 0.21460003, 0.044200003, 0.04530002,  0.0064999983]*/        
        
        /* maxIdxs  1,1     1,2     1,4
                    3,1     3,3     3,4
                    4,1     4,2     4,5  */        
        
        // sum delta * weight and transpose
        // Test: 0.1 * 0.18075174 + 0.2 * 11627263 = 0.0413297 ... 
        Tensor expected = new Tensor(6, 6, 2,
                new float[]{
                            0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
                            0.0f, 0.4f,  0.6f,  0.0f,  0.8f,  0.0f,
                            0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
                            0.0f, 0.24f, 0.0f,  0.26f, 0.28f, 0.0f,
                            0.0f, 0.44f, 0.46f, 0.0f,  0.0f,  0.48f,
                            0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
                            
                            0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
                            0.0f, 0.5f,  0.7f,  0.0f,  0.9f,  0.0f,
                            0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f,
                            0.0f, 0.25f, 0.0f,  0.27f, 0.29f, 0.0f,
                            0.0f, 0.45f, 0.47f, 0.0f,  0.0f,  0.49f,
                            0.0f, 0.0f,  0.0f,  0.0f,  0.0f,  0.0f                            
                           });

        assertArrayEquals(expected.getValues(), actual.getValues(),1e-7f);
    }            
    
    
   /**
    *   
        [0,0]: -0.013200 + -0.013200 + 0.042000 + 0.022000 = 0.0376
        [0,1]: -0.014300 + -0.014400 + 0.063000 + 0.044000 + -0.014300 + 0.023000 = 0.087
        [1,0]: 0.055000 + 0.048000 + -0.025200 + -0.024200 + 0.062000 + 0.032000 = 0.1476
        [1,1]: 0.023100 + 0.060000 + -0.027300 + -0.026400 +  0.052000 + -0.025300 + 0.093000 + 0.064000 + 0.033000 = 0.2461
    * 
    */ 
   @Test
    public void testBackwardFromSingleConvolutionalToSingleConvolutionalChannel() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f
                });

        Tensor filter = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f
                            });

        float[] biases = new float[]{0.0f};

         
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.activationType = ActivationType.LINEAR;
        instance.init();
        instance.filters[0] = filter;
        instance.biases = biases;
             
        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 1);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.init();
        nextLayer.filters[0] = filter;
        nextLayer.biases = biases;        
                           
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
    public void testBackwardFromSingleConvolutionalToTwoConvolutionalChannels() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f
                });

        Tensor filter = new Tensor(3, 3, 
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f
                            });

        float[] biases = new float[]{0.0f};

        Tensor filter2 = new Tensor(3, 3, 2, 
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f,
                              
                              0.2f, 0.4f, 0.6f,
                              -0.22f, -0.24f, -0.26f,
                              0.8f, 1.0f, 0.42f                       
                            });        
        
         
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 2);
        instance.setPrevLayer(inputLayer);
        instance.activationType = ActivationType.LINEAR;
        instance.init();
        instance.filters[0] = filter;
        instance.filters[1] = filter;
        instance.biases = new float[]{0.0f, 0.0f};
             
        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 1);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.init();
        nextLayer.filters[0] = filter2;
        nextLayer.biases = biases;        
                           
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

                                0.0752f, 0.1740f, 0.1788f, 0.1836f, 0.1884f, 0.1766f,
                                0.2952f, 0.4922f, 0.5192f, 0.5462f, 0.5732f, 0.4496f,
                                0.4892f, 0.7622f, 0.7892f, 0.8162f, 0.8432f, 0.6416f,
                                0.6832f, 1.0322f, 1.0592f, 1.0862f, 1.1132f, 0.8336f,
                                0.8772f, 1.3022f, 1.3292f, 1.3562f, 1.3832f, 1.0256f,
                                0.6432f, 0.7122f, 0.7272f, 0.7422f, 0.7572f, 0.4636f                      
                           });


        assertArrayEquals(expected.getValues(), actual.getValues(), 1e-4f); // 0.8432 vs 0.8431999
    }             
    
   // from multiple conv channels to single conv channel
   @Test
    public void testBackwardFromTwoConvolutionalChannelsToSingleConvolutionalChannel() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f
                });

        Tensor filter = new Tensor(3, 3,
                new float[]{
                              0.1f,   0.2f,  0.3f,
                             -0.11f, -0.12f, -0.13f,
                              0.4f,   0.5f,  0.21f
                            });

        float[] biases = new float[]{0.0f};

         
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.activationType = ActivationType.LINEAR;
        instance.init();
        instance.filters[0] = filter;
        instance.biases = biases;
             
        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 2);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.init();
        nextLayer.filters[0] = filter;
        nextLayer.filters[1] = filter;
        nextLayer.biases = new float[]{0.0f, 0.0f};
                           
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
    
   // from multiple conv channels to single conv channel
   @Test
    public void testBackwardFromTwoConvolutionalChannelsToTwoConvolutionalChannels() {
        RandomGenerator.getDefault().initSeed(123);
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6,
                new float[]{ 
                             0.3f, 0.5f, 0.6f, 0.2f, 0.14f, 0.1f,
                            -0.6f, 0.51f, 0.23f, 0.14f, 0.28f, 0.61f,
                            -0.15f, 0.47f, 0.34f, 0.46f, 0.72f, 0.61f,
                             0.43f, 0.34f, 0.62f, 0.31f, -0.25f, 0.17f,
                             0.53f, 0.41f, 0.73f, 0.92f, -0.21f, 0.84f,
                             0.18f, 0.74f, 0.28f, 0.37f, 0.15f, 0.62f
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

         
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 2);
        instance.setPrevLayer(inputLayer);
        instance.activationType = ActivationType.LINEAR;
        instance.init();
        instance.filters[0] = filter;
        instance.biases = biases;
             
        ConvolutionalLayer nextLayer = new ConvolutionalLayer(3, 3, 2);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
        nextLayer.activationType = ActivationType.LINEAR;
        nextLayer.init();
        nextLayer.filters[0] = filter2;
        nextLayer.filters[1] = filter2;
        nextLayer.biases = biases;
                           
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
    
    
    // forward convolution  applied on deltas - same as backward convolution - just experimenting
    @Ignore
    public void testBackwardSingleFilter() {
        InputLayer inputLayer = new InputLayer(6, 6, 1);
        Tensor input = new Tensor(6, 6, // ovo su delte iz sledeceg lejera u prethodnom testu
                                new float[] {
                                            0.11f, 0.12f, 0.13f, 0.14f, 0.15f, 0.16f,
                                            0.21f, 0.22f, 0.23f, 0.24f, 0.25f, 0.26f,
                                            0.31f, 0.32f, 0.33f, 0.34f, 0.35f, 0.36f,
                                            0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f,
                                            0.51f, 0.52f, 0.53f, 0.54f, 0.55f, 0.56f,
                                            0.61f, 0.62f, 0.63f, 0.64f, 0.65f, 0.66f
                                });
        
        float[] biases = new float[] {0.0f};
        
        Tensor filter = new Tensor(3, 3,
                                new float[] {
                                      0.1f,   0.2f,  0.3f,
                                     -0.11f, -0.12f, -0.13f,
                                      0.4f,   0.5f,  0.21f
                                });        

        Tensor expectedOutputs = new Tensor(6, 6,
                                new float[] {
                                                0.12239999f, 0.1989f, 0.2064f, 0.21389998f, 0.2214f, 0.1943f,
                                                0.2264f, 0.3479f, 0.3614f, 0.3749f, 0.3884f, 0.3083f,
                                                0.3224f, 0.4829f, 0.4964f, 0.5099f, 0.5234f, 0.4053f,
                                                0.4184f, 0.6179f, 0.6314f, 0.64489996f, 0.6584f, 0.5023f,
                                                0.5144f, 0.7529f, 0.7664f, 0.7799f, 0.7934f, 0.5993f,
                                                0.1042f, 0.0906f, 0.093f, 0.0954f, 0.0978f, 0.0163f
                                            });
                
        ConvolutionalLayer instance = new ConvolutionalLayer(3, 3, 1);
        instance.setPrevLayer(inputLayer);
        instance.setActivationType(ActivationType.LINEAR);
        instance.init();
        instance.filters[0] = filter;
        instance.biases = biases;
                
        inputLayer.setInput(input);
        instance.forward();
        Tensor actualOutputs = instance.getOutputs();
        
        assertArrayEquals(expectedOutputs.getValues(), actualOutputs.getValues(), 1e-7f);               
    }    
    
    
    
    
    @Ignore
    public void testApplyWeightChanges() {
        System.out.println("applyWeightChanges");
        ConvolutionalLayer instance = null;
        instance.applyWeightChanges();
        fail("The test case is a prototype.");
    }

}