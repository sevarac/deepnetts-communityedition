package deepnetts.net.layers;

import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import deepnetts.util.Tensors;
import deepnetts.util.WeightsInit;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Dense layer tests with various activation functions
 *
 * TODO:
 *  testForwardWith2DPrevLayer i backward takodje
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class DenseLayerTest {

    /**
     * Test of forward pass, of class DenseLayer using Linear
     * activation function. Test if matrix multiplication and bias addition
     * works correctly output = inputs * weights + bias (where * is matrix
     * multiplication (dot product))
     * 
     * Doublechecked with octave: 19.01.19.
     */
    @Test
    public void testForwardWithLinearActivation() {
        // initialize weights with specified random seed
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)

        // input vector for this layer
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10); // weights matrix
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"

        // create prev fc layer with 5 outputs
        DenseLayer prevLayer = new DenseLayer(5);
        prevLayer.setOutputs(input);

        // create instance of layer to test
        DenseLayer instance = new DenseLayer(10, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);
        instance.init(); // init weights structure
        instance.setWeights(weights); // set weight values
        instance.setBiases(new float[]{0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values

        // do the forward pass
        instance.forward();
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();   // "[0.042127118, 0.36987683, 0.10604945, 0.24532129, 0.17567813, 0.34893453, 0.16589889, -0.3487752, 0.09166323, -0.015247092]"
        Tensor expectedOutputs = new Tensor(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);

        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }
    
    /**
    * Doublechecked with octave: 19.01.19.
    */
    @Test
    public void testForwardWithSigmoidActivation() {
        RandomGenerator.getDefault().initSeed(123); // init default random generator with seed that will be used for weights2 (same effect as line above)
        // input vector for this layer
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10);
        WeightsInit.uniform(weights.getValues(), 5); // [0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]

        // create prev fc layer with 5 outputs
        DenseLayer prevLayer = new DenseLayer(5);
        prevLayer.setOutputs(input);

        // create instance of layer to test
        DenseLayer instance = new DenseLayer(10, ActivationType.SIGMOID);
        instance.setPrevLayer(prevLayer);
        instance.init(); // init weights2 structure
        instance.setWeights(weights); // set weights2 values
        instance.setBiases(new float[]{0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values

        // run forward pass
        instance.forward();
        
        // get layer outputs
        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(0.51053022f, 0.59142921f, 0.52648754f, 0.56102458f, 0.54380692f, 0.58635918f, 0.54137987f, 0.41367945f, 0.52289978f, 0.4961883f);
//                                  octave: 0.51053      0.59143      0.52649      0.56102      0.54381      0.58636      0.54138      0.41368      0.52290      0.49619
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }

    /**
     * Doublechecked with octave: 19.01.19.
     */
    @Test
    public void testForwardWithTanhActivation() {
        RandomGenerator.getDefault().initSeed(123); // init default random generator with seed that will be used for weights2 (same effect as line above)
        // input vector for this layer
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10);
        WeightsInit.uniform(weights.getValues(), 5); // [0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]

        // create prev fc layer with 5 outputs
        DenseLayer prevLayer = new DenseLayer(5);
        prevLayer.setOutputs(input);

        // create instance of layer to test
        DenseLayer instance = new DenseLayer(10, ActivationType.TANH);
        instance.setPrevLayer(prevLayer);
        instance.init(); // init weights2 structure
        instance.setWeights(weights); // set weights2 values
        instance.setBiases(new float[]{0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values

        // run forward pass
        instance.forward();
        // get layer outputs
        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(0.0421022217154f, 0.353883947707f, 0.105653667421f, 0.240515590945f, 0.173892848898f, 0.335430293333f, 0.16439350833f, -0.335288915023f, 0.0914073785089f, -0.0152459102444f);
//                                  octave: 0.042102          0.353884         0.105654         0.240516         0.173893         0.335430         0.164394        -0.335289         0.091407          -0.015246
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }

    /**
    * Doublechecked: 19.01.19.
    */    
    @Test
    public void testForwardWithReluActivation() {
        RandomGenerator.getDefault().initSeed(123); // init default random generator with seed that will be used for weights2 (same effect as line above)
        // input vector for this layer
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10);
        WeightsInit.uniform(weights.getValues(), 5); // [0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]

        // create prev fc layer with 5 outputs
        DenseLayer prevLayer = new DenseLayer(5);
        prevLayer.setOutputs(input);

        // create instance of layer to test
        DenseLayer instance = new DenseLayer(10, ActivationType.RELU);
        instance.setPrevLayer(prevLayer);
        instance.init(); // init weights2 structure
        instance.setWeights(weights); // set weights2 values
        instance.setBiases(new float[]{0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values

        // run forward pass
        instance.forward();
        // get layer outputs
        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(0.0421271f, 0.369877f, 0.106049f, 0.245321f, 0.175678f, 0.348935f, 0.165899f, 0.0f, 0.0916632f, 0.0f);
   
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-6f);
    }

    /**
     * Testiram ga prvo sa jednim izlaznim neuronom
     * Doublechecked with octave: 23.01.19.
     */
    @Test
    public void testForward2DInputSingleOutput() {
        // initialize weights with specified random seed
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)

        // input vector for this layer  width:4, height:3
        Tensor input = Tensors.random(3, 4);    // random input matrix [0.72317415, 0.23724389, 0.99089885, 0.30157375, 0.2532931, 0.57412946, 0.60880035, 0.2588815, 0.80586946, 0.6223695, 0.87541276, 0.5492985]
                
        // create prev fc layer with 5 outputs
        InputLayer prevLayer = new InputLayer(4, 3);
        prevLayer.setOutputs(input);

        // create instance of layer to test
        DenseLayer instance = new DenseLayer(1, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);
        instance.init(); // init weights structure
        
        // weights for dense layer: prevLayer.width, prevLayer.height, prevLayer.depth, width
        Tensor weights = new Tensor(4, 3, 1, 1); // weights matrix  
        WeightsInit.uniform(weights.getValues(), 12); // [-0.02413708, -0.25080326, -0.23727596, -0.24853897, -0.21809235, -0.2362793, -0.09346612, -0.043609187, 0.24941927, 0.21615475, 0.102702856, 0.043361217]
                
        instance.setWeights(weights); // set weight values
        instance.setBiases(new float[]{0.1f}); // set bias values

        // do the forward pass
        instance.forward();
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();   // [-0.2886533504537852]
        Tensor expectedOutputs = new Tensor(-0.28865337f);
                                 // octave: -0.28865

        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }
    
    /**
     * 2d input, 2 neurons in dense layer
     * Doublechecked with octave: 23.01.19.
     */
    @Test
    public void testForward2DInputTwoOutputs() {
        // initialize weights with specified random seed
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)

        // input vector for this layer  width:4, height:3
        Tensor input = Tensors.random(3, 4);    // random input matrix [0.72317415, 0.23724389, 0.99089885, 0.30157375, 0.2532931, 0.57412946, 0.60880035, 0.2588815, 0.80586946, 0.6223695, 0.87541276, 0.5492985]
                
        // create prev fc layer with 5 outputs
        InputLayer prevLayer = new InputLayer(4, 3);
        prevLayer.setOutputs(input);

        // create instance of layer to test
        DenseLayer instance = new DenseLayer(2, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);
        instance.init(); // init weights structure
        
        // trebao bh prvo da ga testiram sa jednim izlaznim neuronom
        // weights for dense layer: prevLayer.width, prevLayer.height, prevLayer.depth, width
        Tensor weights = new Tensor(4, 3, 1, 2); // weights matrix  
        WeightsInit.uniform(weights.getValues(), 12); // [-0.059757575, -7.870793E-5, -0.13953678, 0.27760875, -0.20046058, 0.17424765, -0.13831584, 0.16441566, -0.02212295, -0.28034917, -0.035160214, -0.15328945, -0.16662017, -0.01334855, 0.06594038, -0.20663363, -0.26935822, -0.034936756, 0.109059215, -0.22400141, 0.10629755, 0.20143756, -0.24522439, -0.19821966]
                
        instance.setWeights(weights); // set weight values
        instance.setBiases(new float[]{0.1f, 0.2f}); // set bias values

        // do the forward pass
        instance.forward();
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor(-0.23064378f, -0.14301854f);
                                // octave:  -0.23064,     -0.14302
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }    

    @Test
    public void testForward3DInputSingleOutput() {
        // initialize weights with specified random seed
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)

        // input vector for this layer  width:4, height:3
        Tensor input = Tensors.random(3, 4, 2);    // random input matrix [0.72317415, 0.23724389, 0.99089885, 0.30157375, 0.2532931, 0.57412946, 0.60880035, 0.2588815, 0.80586946, 0.6223695, 0.87541276, 0.5492985, 0.7160485, 0.16221565, 0.071917, 0.6818127, 0.79626095, 0.26765352, 0.57871693, 0.24259669, 0.9081256, 0.60227644, 0.14891458, 0.21662551]
                
        // create prev fc layer with 5 outputs
        InputLayer prevLayer = new InputLayer(4, 3, 2);
        prevLayer.setOutputs(input);

        // create instance of layer to test
        DenseLayer instance = new DenseLayer(1, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);
        instance.init(); // init weights structure
        
        // trebao bh prvo da ga testiram sa jednim izlaznim neuronom
        // weights for dense layer: prevLayer.width, prevLayer.height, prevLayer.depth, width
        Tensor weights = new Tensor(4, 3, 2, 1); // weights matrix  
        WeightsInit.uniform(weights.getValues(), 24); // [-0.108392015, -0.11781825, -0.009438843, 0.04662688, -0.14611204, -0.19046502, -0.024704024, 0.077116504, -0.1583929, 0.07516374, 0.14243786, -0.17339982, -0.14016245, 0.18240763, -0.040653482, -0.025770023, 0.17933364, 0.07912485, -0.034144267, 0.07201438, 0.13413312, 0.052575663, 0.06653626, 0.03049782]
                
        instance.setWeights(weights); // set weight values
        instance.setBiases(new float[]{0.1f}); // set bias values

        // do the forward pass
        instance.forward();
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();   // [-0.14187025]  
        Tensor expectedOutputs = new Tensor(-0.14187025f); // octave kaze 0.058130

        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }    
    
    @Test       // ovaj izgleda nije doublechecked
    public void testForward3DInputTwoOutputs() {
        // initialize weights with specified random seed
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)

        // input vector for this layer  width:4, height:3
        Tensor input = Tensors.random(3, 4, 2);    // random input matrix [0.72317415, 0.23724389, 0.99089885, 0.30157375, 0.2532931, 0.57412946, 0.60880035, 0.2588815, 0.80586946, 0.6223695, 0.87541276, 0.5492985, 0.7160485, 0.16221565, 0.071917, 0.6818127, 0.79626095, 0.26765352, 0.57871693, 0.24259669, 0.9081256, 0.60227644, 0.14891458, 0.21662551]
                
        // create prev fc layer with 5 outputs
        InputLayer prevLayer = new InputLayer(4, 3, 2);
        prevLayer.setOutputs(input);

        // create instance of layer to test
        DenseLayer instance = new DenseLayer(2, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);
        instance.init(); // init weights structure
        
        // trebao bh prvo da ga testiram sa jednim izlaznim neuronom
        // weights for dense layer: prevLayer.width, prevLayer.height, prevLayer.depth, width
        Tensor weights = new Tensor(4, 3, 2, 2); // weights matrix  
        WeightsInit.uniform(weights.getValues(), 48); // [-0.032339804, 0.07949282, -0.08974095, -0.024059728, -0.13662373, -0.090818, -0.13522792, 0.032500148, 0.09538056, -0.029679954, 0.06930688, 0.09398733, 0.12744209, 0.09383395, 0.09409064, -0.051186137, 0.0041013956, 0.0826029, 0.027157366, 0.03552276, -0.078440405, -0.042539023, -0.056820266, -0.095608585, 0.056304574, 0.067483634, -0.020708613, -0.020137347, 0.13437173, -0.018618405, 0.06507628, -0.07393739, -0.04758586, -0.062255293, 0.13229671, -0.03867311, -0.0042365193, -0.060115047, -0.07344663, -0.08496776, 0.056355953, -0.012163326, -0.08099257, 0.1287122, -0.11224161, 0.08320823, 0.019138098, 0.0032341331]
                
        instance.setWeights(weights); // set weight values
        instance.setBiases(new float[]{0.1f, 0.2f}); // set bias values

        // do the forward pass
        instance.forward();
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs(); 
        Tensor expectedOutputs = new Tensor(0.15495503f, 0.2643498f); // octave kaze -0.1549549936880784, 0.2643497903952628    Zasto ovde minus????

        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }        
    
    /**
     * Test of backward method, of class DenseLayer using linear
     * activation function. Checks if deltas and delta weights are calculated correctly.
     *
     * kad ide backwrd treba dda trasponuje matricu tezina i da je pomnozi sa
     * deltama iz sledeceg lejera d1 = d2 * W'  gde je W' transponovana matrica tezina
     *
     * Doublechecked with octave 23.01.19.
     */
    @Test
    public void testBackwardLinear() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f); // input vector for this layer

        Tensor weights2 = new Tensor(5, 10);
        WeightsInit.uniform(weights2.getValues(), 5); // [0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]

        Tensor weights1 = new Tensor(5, 5);
        WeightsInit.uniform(weights1.getValues(), 5);    // [-0.25812685, -0.020679474, 0.102154374, -0.32011545, -0.41728795, -0.05412379, 0.16895384, -0.3470215, 0.16467547, 0.31206572, -0.37989998, -0.30708057, 0.39963514, -0.08906731, -0.056459278, 0.39290035, 0.17335385, -0.07480636, 0.15777558, 0.29387093, 0.115187526, 0.14577365, 0.0668174, -0.4196531, -0.10020122]

        Tensor nextDeltas = new Tensor(10);
        nextDeltas.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);

        DenseLayer prevLayer = new DenseLayer(5); // feeds input into tested layer
        prevLayer.setOutputs(inputs);

        DenseLayer instance = new DenseLayer(5, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);
        instance.init();
        instance.setWeights(weights1);
        instance.setOptimizerType(OptimizerType.SGD);
        instance.setBiases(new float[]{0.1f, 0.2f, 0.3f, 0.11f, 0.12f}); // set bias values

        DenseLayer nextLayer = new DenseLayer(10);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
        nextLayer.init();
        nextLayer.setWeights(weights2);
        nextLayer.setDeltas(nextDeltas);

        instance.forward();  // da bi izracunao output
        instance.backward(); // deltas

        Tensor result = instance.deltas;
        Tensor expResult = new Tensor(5); 
        expResult.setValues(0.02364707f, 0.09271421f, 0.04896196f, 0.29104635f, 0.37890932f);
                 // octave: 0.023647     0.092714     0.048962     0.291046     0.378909
        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-7f);

        /* octave
        deltaWeights=
  -0.00023647  -0.00092714  -0.00048962  -0.00291046  -0.00378909
  -0.00047294  -0.00185428  -0.00097924  -0.00582093  -0.00757819
  -0.00070941  -0.00278143  -0.00146886  -0.00873139  -0.01136728
  -0.00094588  -0.00370857  -0.00195848  -0.01164185  -0.01515637
  -0.00118235  -0.00463571  -0.00244810  -0.01455232  -0.01894547                
        */
        Tensor deltaWeights = instance.getDeltaWeights();
        Tensor expDeltaWeights = new Tensor(-0.000236471f, -0.000927142f, -0.00048962f, -0.00291046f, -0.00378909f, -0.000472941f, -0.00185428f, -0.000979239f, -0.00582093f, -0.00757819f, -0.000709412f, -0.00278143f, -0.00146886f, -0.00873139f, -0.0113673f, -0.000945883f, -0.00370857f, -0.00195848f, -0.0116419f, -0.0151564f, -0.00118235f, -0.00463571f, -0.0024481f, -0.0145523f, -0.0189455f);
        
        assertArrayEquals(expDeltaWeights.getValues(), deltaWeights.getValues(), 1e-7f);
        
        // octave biases:  -0.0023647  -0.0092714  -0.0048962  -0.0291046  -0.0378909
        
        float[] deltaBiases = instance.getDeltaBiases();
        float[] expDeltaBiases = new float[] {  -0.0023647f, -0.0092714f, -0.0048962f, -0.0291046f, -0.0378909f};       
                             // octave:         -0.0023647   -0.0092714   -0.0048962   -0.0291046   -0.0378909
         assertArrayEquals(expDeltaBiases, deltaBiases, 1e-7f);          
         
    }

    /**
     * Test of backward method, of class FullyConnectedLayer using sigmoid
     * activation function.
     *
     * Tests only when prev layer is also FullyConnectd It should test also when
     * prev layer is maxpooling or convolutional
     * 
     * Double checked with octave 23.1.19.
     */
    @Test
    public void testBackwardSigmoid() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)
        
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f); // input vector for this layer

        Tensor weights2 = new Tensor(5, 10);
        WeightsInit.uniform(weights2.getValues(), 5); // [0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]

        Tensor weights1 = new Tensor(5, 5);
        WeightsInit.uniform(weights1.getValues(), 5);    // [-0.25812685, -0.020679474, 0.102154374, -0.32011545, -0.41728795, -0.05412379, 0.16895384, -0.3470215, 0.16467547, 0.31206572, -0.37989998, -0.30708057, 0.39963514, -0.08906731, -0.056459278, 0.39290035, 0.17335385, -0.07480636, 0.15777558, 0.29387093, 0.115187526, 0.14577365, 0.0668174, -0.4196531, -0.10020122]

        Tensor nextDeltas = new Tensor(10);
        nextDeltas.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);

        // previous layer
        DenseLayer prevLayer = new DenseLayer(5); // feeds input into tested layer
        prevLayer.setOutputs(inputs);

        // layer to test
        DenseLayer instance = new DenseLayer(5, ActivationType.SIGMOID);
        instance.setPrevLayer(prevLayer);
        instance.init();
        instance.setWeights(weights1);
        instance.setOptimizerType(OptimizerType.SGD);
        instance.setBiases(new float[]{0.1f, 0.2f, 0.3f, 0.11f, 0.12f}); // set bias values
    
        // next layer
        DenseLayer nextLayer = new DenseLayer(10);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
        nextLayer.init();
        nextLayer.setWeights(weights2);
        nextLayer.setDeltas(nextDeltas);
        
        instance.forward();  // da bi izracunao output
        instance.backward(); // deltas

        Tensor result = instance.deltas;
        Tensor expResult = new Tensor(5); // "[0.005872122, 0.022724332, 0.011843424, 0.07269054, 0.093866885]"
        expResult.setValues(0.00587212f, 0.02272433f, 0.01184342f, 0.07269055f, 0.09386688f);
              // octave:    0.0058721    0.0227243    0.0118434    0.0726905    0.0938669

        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-8f);

/* octave
   delta_weights =

  -0.000058721  -0.000227243  -0.000118434  -0.000726905  -0.000938669
  -0.000117442  -0.000454487  -0.000236868  -0.001453811  -0.001877338
  -0.000176164  -0.000681730  -0.000355303  -0.002180716  -0.002816006
  -0.000234885  -0.000908973  -0.000473737  -0.002907622  -0.003754675
  -0.000293606  -0.001136217  -0.000592171  -0.003634527  -0.004693344
        
        */        
        
        Tensor deltaWeights = instance.getDeltaWeights();
        Tensor expDeltaWeights = new Tensor(-5.87212395072e-05f, -0.000227243306352f, -0.000118434237896f, -0.000726905494703f, -0.000938668826303f, -0.000117442479014f, -0.000454486612705f, -0.000236868475791f, -0.00145381098941f, -0.00187733765261f, -0.000176163722897f, -0.000681729935988f, -0.000355302722511f, -0.00218071653827f, -0.00281600654885f, -0.000234884958029f, -0.000908973225409f, -0.000473736951583f, -0.00290762197881f, -0.00375467530521f, -0.000293606193161f, -0.00113621651483f, -0.000592171180654f, -0.00363452741935f, -0.00469334406158f);

        assertArrayEquals(expDeltaWeights.getValues(), deltaWeights.getValues(), 1e-9f);
        
/*
      octave delta biases:     -0.00058721  -0.00227243  -0.00118434  -0.00726905  -0.00938669
        */
        float[] deltaBiases = instance.getDeltaBiases();
        float[] expDeltaBiases = new float[] {  -0.00058721f, -0.00227243f, -0.00118434f, -0.00726905f, -0.00938669f };       
                             // octave:         -0.00058721  -0.00227243  -0.00118434  -0.00726905  -0.00938669
         assertArrayEquals(expDeltaBiases, deltaBiases, 1e-7f);          
        
    }

    /**
     * Double checked with octave 23.1.19.
     */    
    @Test
    public void testBackwardTanh() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f); // input vector for this layer

        Tensor weights2 = new Tensor(5, 10);
        WeightsInit.uniform(weights2.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"

        Tensor weights1 = new Tensor(5, 5);
        WeightsInit.uniform(weights1.getValues(), 5);    // "[-0.25812685, -0.020679474, 0.102154374, -0.32011545, -0.41728795, -0.05412379, 0.16895384, -0.3470215, 0.16467547, 0.31206572, -0.37989998, -0.30708057, 0.39963514, -0.08906731, -0.056459278, 0.39290035, 0.17335385, -0.07480636, 0.15777558, 0.29387093, 0.115187526, 0.14577365, 0.0668174, -0.4196531, -0.10020122]"

        Tensor nextDeltas = new Tensor(10);
        nextDeltas.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);

        DenseLayer prevLayer = new DenseLayer(5); // not used for anything just dummy to prevent npe in init
        prevLayer.setOutputs(inputs);

        DenseLayer instance = new DenseLayer(5, ActivationType.TANH);
        instance.setPrevLayer(prevLayer);
        instance.init();
        instance.setOptimizerType(OptimizerType.SGD);
        instance.setWeights(weights1);
        instance.setBiases(new float[]{0.1f, 0.2f, 0.3f, 0.11f, 0.12f}); // set bias values

        DenseLayer nextLayer = new DenseLayer(10);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
        nextLayer.init();
        nextLayer.setWeights(weights2);
        nextLayer.setDeltas(nextDeltas);

        instance.forward();
        instance.backward();
        
        Tensor result = instance.deltas;
        Tensor expResult = new Tensor(5); 
        expResult.setValues(0.02302119f, 0.08572333f, 0.04300185f, 0.28991194f, 0.36538888f);
    // octave deltas:       0.023021     0.085723     0.043002     0.289912     0.365389
        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-7f);

/*
 octave delta_weights =

  -0.00023021  -0.00085723  -0.00043002  -0.00289912  -0.00365389
  -0.00046042  -0.00171447  -0.00086004  -0.00579824  -0.00730778
  -0.00069064  -0.00257170  -0.00129006  -0.00869736  -0.01096167
  -0.00092085  -0.00342893  -0.00172007  -0.01159648  -0.01461556
  -0.00115106  -0.00428617  -0.00215009  -0.01449560  -0.01826944        
        
*/                
        Tensor deltaWeights = instance.getDeltaWeights();
        Tensor expDeltaWeights = new Tensor(-0.00023021194604f, -0.000857233286545f, -0.00043001848101f, -0.0028991194114f, -0.00365388885605f, -0.000460423892081f, -0.00171446657309f, -0.00086003696202f, -0.0057982388228f, -0.0073077777121f, -0.000690635855273f, -0.0025716999235f, -0.00129005547507f, -0.0086973584502f, -0.0109616668404f, -0.000920847784162f, -0.00342893314618f, -0.00172007392404f, -0.0115964776456f, -0.0146155554242f, -0.00115105971305f, -0.00428616636886f, -0.00215009237301f, -0.014495596841f, -0.018269444008f);

        assertArrayEquals(expDeltaWeights.getValues(), deltaWeights.getValues(), 1e-8f);
        
/*
      octave delta biases:      -0.0023021  -0.0085723  -0.0043002  -0.0289912  -0.0365389
        */
        float[] deltaBiases = instance.getDeltaBiases();
        float[] expDeltaBiases = new float[] {    -0.0023021f, -0.0085723f, -0.0043002f, -0.0289912f, -0.0365389f };       
                             // octave:           -0.0023021  -0.0085723  -0.0043002  -0.0289912  -0.0365389
         assertArrayEquals(expDeltaBiases, deltaBiases, 1e-7f);           
    }

    // not double checked...
    @Test
    public void testBackwardRelu() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f); // input vector for this layer

        Tensor weights2 = new Tensor(5, 10);
        WeightsInit.uniform(weights2.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"

        Tensor weights1 = new Tensor(5, 5);
        WeightsInit.uniform(weights1.getValues(), 5);    // "[-0.25812685, -0.020679474, 0.102154374, -0.32011545, -0.41728795, -0.05412379, 0.16895384, -0.3470215, 0.16467547, 0.31206572, -0.37989998, -0.30708057, 0.39963514, -0.08906731, -0.056459278, 0.39290035, 0.17335385, -0.07480636, 0.15777558, 0.29387093, 0.115187526, 0.14577365, 0.0668174, -0.4196531, -0.10020122]"

        Tensor nextDeltas = new Tensor(10);
        nextDeltas.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);

        DenseLayer prevLayer = new DenseLayer(5); // not used for anything just dummy to prevent npe in init
        prevLayer.setOutputs(inputs);

        DenseLayer instance = new DenseLayer(5, ActivationType.RELU);
        instance.setPrevLayer(prevLayer);
        instance.init();
        instance.setOptimizerType(OptimizerType.SGD);
        instance.setWeights(weights1);
        instance.setBiases(new float[]{0.1f, 0.2f, 0.3f, 0.11f, 0.12f}); // set bias values

        DenseLayer nextLayer = new DenseLayer(10);
        nextLayer.setPrevLayer(instance);
        instance.setNextlayer(nextLayer);
        nextLayer.init();
        nextLayer.setWeights(weights2);
        nextLayer.setDeltas(nextDeltas);

        instance.forward();
        instance.backward();

        Tensor result = instance.deltas;
        Tensor expResult = new Tensor(5); // "[0.005872122, 0.022724332, 0.011843424, 0.07269054, 0.093866885]"
        expResult.setValues(0.02364707f, 0.09271421f, 0.04896196f, 0.0f, 0.37890932f);

        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-7f);

        Tensor deltaWeights = instance.getDeltaWeights();
        Tensor expDeltaWeights = new Tensor(-0.000236470702834f, -0.000927142142164f, -0.000489619605089f, -0.0f, -0.00378909325285f, -0.000472941405668f, -0.00185428428433f, -0.000979239210177f, -0.0f, -0.00757818650571f, -0.00070941212612f, -0.00278142649557f, -0.00146885885175f, -0.0f, -0.0113672800409f, -0.000945882811336f, -0.00370856856866f, -0.00195847842035f, -0.0f, -0.0151563730114f, -0.00118235349655f, -0.00463571064174f, -0.00244809798896f, -0.0f, -0.018945465982f);

        assertArrayEquals(expDeltaWeights.getValues(), deltaWeights.getValues(), 1e-8f);

    }

    /**
     * Test of applyWeightChanges method, of class DenseLayer. in
     * online mode.
     * todo: test batch mode, da li pamti prev weights2, da li
     * resetuje deltaWeights i deltaBiases na nulu
     */
    @Test
    public void testApplyWeightChanges() {
        RandomGenerator.getDefault().initSeed(123);

        DenseLayer prevLayer = new DenseLayer(5);
        DenseLayer instance = new DenseLayer(10);
        instance.setPrevLayer(prevLayer);
        instance.init();

        Tensor weights = new Tensor(5, 10);
        WeightsInit.initSeed(123);

        Tensor biases = new Tensor(10); //  ? zasto ovde isod opet randomize kad gore imam uniform
        WeightsInit.uniform(weights.getValues(), 5); //  0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501
        WeightsInit.randomize(biases.getValues()); // "[-0.2885946, -0.023120344, 0.114212096, -0.35789996, -0.46654212, -0.060512245, 0.18889612, -0.38798183, 0.18411279, 0.34890008]"
        instance.setWeights(weights);
        instance.setBiases(biases.getValues());
        instance.deltaBiases = new float[10];
        instance.deltaWeights = new Tensor(5, 10);
        WeightsInit.randomize(instance.deltaWeights.getValues()); // "[-0.4247411, -0.3433265, 0.44680566, -0.09958029, -0.063123405, 0.43927592, 0.19381553, -0.083636045, 0.17639846, 0.32855767, 0.12878358, 0.1629799, 0.07470411, -0.46918643, -0.11202836, 0.2753712, -0.31087178, -0.08334535, -0.47327846, -0.3146028, -0.46844327, 0.112583816, 0.33040798, -0.10281438, 0.24008608, 0.32558167, 0.44147235, 0.32505035, 0.32593954, -0.17731398, 0.014207661, 0.28614485, 0.09407586, 0.123054445, -0.27172554, -0.14735949, -0.19683117, -0.33119786, 0.19504476, 0.23377019, -0.07173675, -0.06975782, 0.46547735, -0.06449604, 0.22543085, -0.25612664, -0.16484225, -0.21565866, 0.45828927, -0.13396758]"
        WeightsInit.randomize(instance.deltaBiases); // "[-0.014675736, -0.20824462, -0.2544266, -0.29433697, 0.19522274, -0.042135, -0.2805665, 0.44587213, -0.38881636, 0.2882418]"

        instance.applyWeightChanges();
        Tensor expectedWeights = new Tensor(-0.22512805f, -0.57834274f, 0.8858789f, -0.27705812f, -0.28378475f, 0.50557935f, 0.29112953f, -0.29929897f, 0.44997644f, 0.4380083f, 0.46456295f, 0.20707384f, 0.26794374f, -0.7713099f, -0.49491742f, 0.4379894f, -0.045887947f, -0.29116237f, -0.4028719f, -0.5448313f, -0.10340464f, 0.20406264f, 0.016387641f, -0.35627222f, 0.6651356f, 0.28818867f, 0.05292958f, -0.04253599f, -0.059095383f, -0.5151812f, -0.35183465f, 0.14134777f, 0.026516795f, 0.5094531f, 0.063139975f, 0.011747062f, -0.12965626f, -0.77672803f, 0.1024687f, 0.23364827f, -0.2879062f, 0.36031187f, 0.15492517f, 0.20544726f, 0.011152849f, -0.0014150143f, -0.19911501f, -0.64997375f, 0.40381932f, -0.3714426f);
        Tensor expectedBiases = new Tensor(10);
        expectedBiases.setValues(-0.30327034f, -0.23136496f, -0.1402145f, -0.65223693f, -0.27131938f, -0.10264724f, -0.09167038f, 0.0578903f, -0.20470357f, 0.63714188f); // 0.4507922, 0.35359812, 0.25770348, -0.39445835, 0.7433155, 0.7244545, -0.07723093, 0.26560563, -0.23044652, 0.2704906

        assertArrayEquals(weights.getValues(), expectedWeights.getValues(), 1e-7f);
        assertArrayEquals(biases.getValues(), expectedBiases.getValues(), 1e-7f);
    }
}
