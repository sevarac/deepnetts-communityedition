package deepnetts.net.layers;

import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import deepnetts.util.WeightsInit;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class OutputLayerTest {
    
    /**
     * Tests forward pass using sigmoid activation function. 
     * Doublechecked with octave: 21.01.19.
     */
    @Test
    public void testForwardSigmoid() {
        
        RandomGenerator.getDefault().initSeed(123);         // initialize weights using specified random seed
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f); // input vector for this layer (output for previous layer)
        Tensor weights = new Tensor(5, 10); 
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]" 

       Tensor expectedOutputs = new Tensor( 0.51053022f, 0.59142921f, 0.52648754f, 0.56102458f, 0.54380692f, 0.58635918f, 0.54137987f, 0.41367945f, 0.52289978f, 0.4961883f );
                              // octave:    0.51053      0.59143      0.52649      0.56102      0.54381      0.58636      0.54138      0.41368      0.52290      0.49619
        // create prev fc layer with 5 outputs
        DenseLayer prevLayer = new DenseLayer(5);        
        prevLayer.setOutputs(input); // and set its ouput that will be used as input for next layer
                
        // create instance of layer to test
        OutputLayer instance = new OutputLayer(10);
        instance.setPrevLayer(prevLayer);        
        instance.init(); // init weights structures
        instance.setWeights(weights); // set weights values
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        
        // run forward pass
        instance.forward();
        
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();
        
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }

    /**
     * Doublechecked with octave: 21.01.19.
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
        OutputLayer instance = new OutputLayer(10, ActivationType.LINEAR);
        instance.setPrevLayer(prevLayer);        
        instance.init(); // init weights structure
        instance.setWeights(weights); // set weights2 values
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        
        // do the forward pass
        instance.forward();        
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();   // "[0.042127118, 0.36987683, 0.10604945, 0.24532129, 0.17567813, 0.34893453, 0.16589889, -0.3487752, 0.09166323, -0.015247092]"
        Tensor expectedOutputs = new Tensor( 0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);

        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }    
    
    /**
     * Doublechecked with octave: 21.01.19.
     */
    @Test
    public void testForwardWithTanhActivation() {
        
        RandomGenerator.getDefault().initSeed(123);         // initialize weights using specified random seed
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f); // input vector for this layer (output for previous layer)
        Tensor weights = new Tensor(5, 10); 
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]" 
        
        // create prev fc layer with 5 outputs
        DenseLayer prevLayer = new DenseLayer(5);        
        prevLayer.setOutputs(input); // and set its ouput that will be used as input for next layer
                
        // create instance of layer to test
        OutputLayer instance = new OutputLayer(10, ActivationType.TANH);
        instance.setPrevLayer(prevLayer);        
        instance.init(); // init weights structures
        instance.setWeights(weights); // set weights values
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        
        // run forward pass
        instance.forward();
        
        // get layer outpputs
        Tensor actualOutputs = instance.getOutputs();
        Tensor expectedOutputs = new Tensor( 0.0421022217154f,  0.353883947707f,  0.105653667421f,  0.240515590945f,  0.173892848898f,  0.335430293333f,  0.16439350833f,  -0.335288915023f,  0.0914073785089f,  -0.0152459102444f);
                               // octave:    0.042102           0.353884          0.105654          0.240516          0.173893          0.335430          0.164394         -0.335289          0.091407           -0.015246
        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }    
    
    /**
     * These values are for backward pass for MSE loss function.
     */
    @Test
    public void testBackwardSigmoidMseLoss() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights (same effect as line above)
        
        // input vector for this layer
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10); // weights from previous layer
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"
        Tensor outputErrors = new Tensor(10);
        outputErrors.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);
                
        DenseLayer prevLayer = new DenseLayer(5); // not used for anything just dummy to prevent npe in init      
        prevLayer.setOutputs(inputs);
        
        OutputLayer instance = new OutputLayer(10);
        instance.setLossType(LossType.MEAN_SQUARED_ERROR);
        instance.setPrevLayer(prevLayer);               
        instance.init();
        instance.setOptimizerType(OptimizerType.SGD);
        instance.setWeights(weights);
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        instance.forward(); // derivatives are calculated using outputs | outputs : "[0.51053023, 0.59142923, 0.5264875, 0.5610246, 0.5438069, 0.5863592, 0.5413798, 0.41367948, 0.52289975, 0.4961883]"          
        instance.setOutputErrors(outputErrors.getValues());

        instance.backward();
                
        Tensor result = instance.getDeltas(); //   [0.01052711, 0.08937729, 0.026437959, 0.060416743, 0.043582395, 0.08463131, 0.04119066, -0.084595, 0.022867741, -0.0038115513]
        Tensor expResult = new Tensor(10); // [0.01052711  0.08937729  0.02643796   0.06041675   0.0435824    0.08463131  0.04119066  -0.084595  0.02286774   -0.00381155] from num py test
        expResult.setValues( 0.01052711f,  0.08937729f,  0.02643796f,  0.06041675f,  0.0435824f,   0.08463131f,  0.04119066f, -0.084595f, 0.02286774f, -0.00381155f);
                  // octave: 0.0105271     0.0893773     0.0264380     0.0604167     0.0435824     0.0846313     0.0411907    -0.0845950  0.0228677    -0.0038116
        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-8f);        
        
        /*
  -0.000105271  -0.000893773  -0.000264380  -0.000604167  -0.000435824  -0.000846313  -0.000411907   0.000845950  -0.000228677   0.000038116
  -0.000210542  -0.001787546  -0.000528759  -0.001208335  -0.000871648  -0.001692626  -0.000823813   0.001691900  -0.000457355   0.000076231
  -0.000315813  -0.002681319  -0.000793139  -0.001812502  -0.001307472  -0.002538939  -0.001235720   0.002537850  -0.000686032   0.000114347
  -0.000421084  -0.003575092  -0.001057518  -0.002416670  -0.001743296  -0.003385252  -0.001647626   0.003383800  -0.000914710   0.000152462
  -0.000526355  -0.004468864  -0.001321898  -0.003020837  -0.002179120  -0.004231566  -0.002059533   0.004229750  -0.001143387   0.000190578                
        */
        
        Tensor deltaWeights = instance.getDeltaWeights();
        Tensor expDeltaWeights = new Tensor(-1.05271092E-4f, -8.93772906E-4f, -2.64379592E-4f, -6.04167468E-4f, -4.35823960E-4f, -8.46313123E-4f, -4.11906628E-4f, 8.45950039E-4f, -2.28677425E-4f, 3.81155098E-5f,  -2.10542184E-4f, -1.78754581E-3f, -5.28759185E-4f, -1.20833494E-3f, -8.71647919E-4f, -1.69262625E-3f, -8.23813255E-4f,  1.69190008E-3f, -4.57354850E-4f,  7.62310196E-5f, -3.15813283E-4f, -2.68131878E-3f, -7.93138797E-4f, -1.81250245E-3f, -1.30747191E-3f, -2.53893943E-3f, -1.23571991E-3f, 2.53785018E-3f, -6.86032292E-4f,  1.14346532E-4f, -4.21084367E-4f, -3.57509162E-3f, -1.05751837E-3f, -2.41666987E-3f, -1.74329584E-3f, -3.38525249E-3f, -1.64762651E-3f,  3.38380016E-3f, -9.14709700E-4f, 1.52462039E-4f, -5.26355451E-4f, -4.46886446E-3f, -1.32189794E-3f, -3.02083730E-3f, -2.17911977E-3f, -4.23156555E-3f, -2.05953311E-3f, 4.22975013E-3f, -1.14338711E-3f, 1.90577546E-4f);
                
        assertArrayEquals(expDeltaWeights.getValues(), deltaWeights.getValues(), 1e-9f);
        
        float[] deltaBiases = instance.getDeltaBiases();
        float[] expDeltaBiases = new float[] {  -0.00105271f, -0.00893773f, -0.00264380f, -0.00604167f, -0.00435824f, -0.00846313f, -0.00411907f, 0.00845950f, -0.00228677f, 0.00038116f};       
                             // octave:         -0.00105271  -0.00893773  -0.00264380  -0.00604167  -0.00435824  -0.00846313  -0.00411907   0.00845950  -0.00228677   0.00038116
         assertArrayEquals(expDeltaBiases, deltaBiases, 1e-7f);        
        
    }
    
    /**
     * Doublechecked with octave: 22.01.19.
     */    
    @Test
    public void testBackwardTanhMseLoss() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights (same effect as line above)
        
        // input vector for this layer
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10); // weights from previous layer
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"
        Tensor outputErrors = new Tensor(10);
        outputErrors.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);
                
        DenseLayer prevLayer = new DenseLayer(5); // not used for anything just dummy to prevent npe in init      
        prevLayer.setOutputs(inputs);
        
        OutputLayer instance = new OutputLayer(10, ActivationType.TANH);
        instance.setLossType(LossType.MEAN_SQUARED_ERROR);
        instance.setPrevLayer(prevLayer);               
        instance.init();
        instance.setOptimizerType(OptimizerType.SGD);
        instance.setWeights(weights);
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        instance.forward(); // derivatives are calculated using outputs | outputs : "[0.51053023, 0.59142923, 0.5264875, 0.5610246, 0.5438069, 0.5863592, 0.5413798, 0.41367948, 0.52289975, 0.4961883]"          
        instance.setOutputErrors(outputErrors.getValues());

        instance.backward();
                                            
        Tensor result = instance.getDeltas(); 
        Tensor expResult = new Tensor(10); 
        expResult.setValues( 0.04205245f, 0.32355571f, 0.10486565f, 0.23113f, 0.17036584f, 0.30967469f, 0.16141546f, -0.30956639f, 0.09089737f, -0.01524355f);
            // octave:       0.042052     0.323556     0.104866     0.231130  0.170366     0.309675     0.161415     -0.309566     0.090897     -0.015244
        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-7f);        
        
/*
Octave delta weights: 
  -0.00042052  -0.00323556  -0.00104866  -0.00231130  -0.00170366  -0.00309675  -0.00161415   0.00309566  -0.00090897   0.00015244
  -0.00084105  -0.00647111  -0.00209731  -0.00462260  -0.00340732  -0.00619349  -0.00322831   0.00619133  -0.00181795   0.00030487
  -0.00126157  -0.00970667  -0.00314597  -0.00693390  -0.00511098  -0.00929024  -0.00484246   0.00928699  -0.00272692   0.00045731
  -0.00168210  -0.01294223  -0.00419463  -0.00924520  -0.00681463  -0.01238699  -0.00645662   0.01238266  -0.00363589   0.00060974
  -0.00210262  -0.01617779  -0.00524328  -0.01155650  -0.00851829  -0.01548373  -0.00807077   0.01547832  -0.00454487   0.00076218               
 */        
        
        Tensor deltaWeights = instance.getDeltaWeights();
        Tensor expDeltaWeights = new Tensor(-0.000420524474547f,  -0.00323555711428f,  -0.0010486565191f,  -0.00231130007696f,  -0.0017036583964f,  -0.0030967469684f,  -0.00161415465798f,  0.00309566398745f,  -0.000908973680493f,  0.000152435460158f,  -0.000841048949094f,  -0.00647111422856f,  -0.0020973130382f,  -0.00462260015392f,  -0.00340731679281f,  -0.00619349393679f,  -0.00322830931597f,  0.0061913279749f,  -0.00181794736099f,  0.000304870920315f,  -0.00126157345497f,  -0.00970667158391f,  -0.00314596963543f,  -0.00693390040308f,  -0.00511097531615f,  -0.00929024113591f,  -0.00484246409421f,  0.00928699219299f,  -0.0027269211092f,  0.000457306391831f,  -0.00168209789819f,  -0.0129422284571f,  -0.0041946260764f,  -0.00924520030784f,  -0.00681463358562f,  -0.0123869878736f,  -0.00645661863193f,  0.0123826559498f,  -0.00363589472197f,  0.000609741840631f,  -0.0021026223414f,  -0.0161777853303f,  -0.00524328251736f,  -0.0115565002126f,  -0.00851829185509f,  -0.0154837346113f,  -0.00807077316965f,  0.0154783197066f,  -0.00454486833474f,  0.000762177289431f);
                
        assertArrayEquals(expDeltaWeights.getValues(), deltaWeights.getValues(), 1e-9f);
        
        // octave:  -0.0042052  -0.0323556  -0.0104866  -0.0231130  -0.0170366  -0.0309675  -0.0161415   0.0309566  -0.0090897   0.0015244
        float[] deltaBiases = instance.getDeltaBiases();
        float[] expDeltaBiases = new float[] { -0.0042052f,  -0.0323556f,  -0.0104866f,  -0.0231130f,  -0.0170366f,  -0.0309675f,  -0.0161415f,   0.0309566f,  -0.0090897f,   0.0015244f};
        assertArrayEquals(expDeltaBiases, deltaBiases, 1e-7f);        
    }    

    /**
     * Doublechecked with octave: 22.01.19.
     */
    @Test
    public void testBackwardLinearMseLoss() {
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights (same effect as line above)
        
        // input vector for this layer
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10); // weights from previous layer
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"
        Tensor outputErrors = new Tensor(10);
        outputErrors.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);
                
        DenseLayer prevLayer = new DenseLayer(5); // not used for anything just dummy to prevent npe in init      
        prevLayer.setOutputs(inputs);
        
        OutputLayer instance = new OutputLayer(10);
        instance.setActivationType(ActivationType.LINEAR);
        instance.setLossType(LossType.MEAN_SQUARED_ERROR);
        instance.setPrevLayer(prevLayer);               
        instance.init();
        instance.setOptimizerType(OptimizerType.SGD);
        instance.setWeights(weights);
        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
        instance.forward(); 
        instance.setOutputErrors(outputErrors.getValues());

        instance.backward();
                
        Tensor result = instance.getDeltas(); //   [0.04212712, 0.3698768, 0.10604945, 0.24532129, 0.17567812, 0.34893453, 0.16589892, -0.34877524, 0.09166324, -0.01524709]
        Tensor expResult = new Tensor(10); 
        expResult.setValues( 0.04212712f,  0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);

        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-8f);        
        
/* octave:
  -0.00042127f,  -0.00369877f,  -0.00106049f,  -0.00245321f,  -0.00175678f,  -0.00348935f,  -0.00165899f,   0.00348775f,  -0.00091663f,   0.00015247f,
  -0.00084254f,  -0.00739754f,  -0.00212099f,  -0.00490643f,  -0.00351356f,  -0.00697869f,  -0.00331798f,   0.00697550f,  -0.00183326f,   0.00030494f,
  -0.00126381f,  -0.01109630f,  -0.00318148f,  -0.00735964f,  -0.00527034f,  -0.01046804f,  -0.00497697f,   0.01046326f,  -0.00274990f,   0.00045741f,
  -0.00168508f,  -0.01479507f,  -0.00424198f,  -0.00981285f,  -0.00702712f,  -0.01395738f,  -0.00663596f,   0.01395101f,  -0.00366653f,   0.00060988f,
  -0.00210636f,  -0.01849384f,  -0.00530247f,  -0.01226606f,  -0.00878391f,  -0.01744673f,  -0.00829495f,   0.01743876f,  -0.00458316f,   0.00076235         
*/        
        Tensor deltaWeights = instance.getDeltaWeights();
        Tensor expDeltaWeights = new Tensor(
  -0.00042127f,  -0.00369877f,  -0.00106049f,  -0.00245321f,  -0.00175678f,  -0.00348935f,  -0.00165899f,   0.00348775f,  -0.00091663f,   0.00015247f,
  -0.00084254f,  -0.00739754f,  -0.00212099f,  -0.00490643f,  -0.00351356f,  -0.00697869f,  -0.00331798f,   0.00697550f,  -0.00183326f,   0.00030494f,
  -0.00126381f,  -0.01109630f,  -0.00318148f,  -0.00735964f,  -0.00527034f,  -0.01046804f,  -0.00497697f,   0.01046326f,  -0.00274990f,   0.00045741f,
  -0.00168508f,  -0.01479507f,  -0.00424198f,  -0.00981285f,  -0.00702712f,  -0.01395738f,  -0.00663596f,   0.01395101f,  -0.00366653f,   0.00060988f,
  -0.00210636f,  -0.01849384f,  -0.00530247f,  -0.01226606f,  -0.00878391f,  -0.01744673f,  -0.00829495f,   0.01743876f,  -0.00458316f,   0.00076235f);
                             
        assertArrayEquals(expDeltaWeights.getValues(), deltaWeights.getValues(), 1e-7f);
        
        float[] deltaBiases = instance.getDeltaBiases();
        float[] expDeltaBiases = new float[] {  -0.0042127f,  -0.0369877f,  -0.0106049f,  -0.0245321f,  -0.0175678f,  -0.0348935f,  -0.0165899f,   0.0348775f,  -0.0091663f, 0.0015247f};        
        assertArrayEquals(expDeltaBiases, deltaBiases, 1e-7f);
    }    
    
    
    @Test
    public void testApplyWeightChanges() {
        OutputLayer instance = new OutputLayer(10);
        DenseLayer prevLayer = new DenseLayer(5);
        instance.setPrevLayer(prevLayer);
        instance.init();
        Tensor weights = new Tensor(5, 10);

        WeightsInit.initSeed(123);
        WeightsInit.uniform(weights.getValues(), 5);    
        Tensor biases = new Tensor(10); // 
        WeightsInit.randomize(weights.getValues()); //  [-0.2885946, -0.023120344, 0.114212096, -0.35789996, -0.46654212, -0.060512245, 0.18889612, -0.38798183, 0.18411279, 0.34890008, -0.4247411, -0.3433265, 0.44680566, -0.09958029, -0.063123405, 0.43927592, 0.19381553, -0.083636045, 0.17639846, 0.32855767, 0.12878358, 0.1629799, 0.07470411, -0.46918643, -0.11202836, 0.2753712, -0.31087178, -0.08334535, -0.47327846, -0.3146028, -0.46844327, 0.112583816, 0.33040798, -0.10281438, 0.24008608, 0.32558167, 0.44147235, 0.32505035, 0.32593954, -0.17731398, 0.014207661, 0.28614485, 0.09407586, 0.123054445, -0.27172554, -0.14735949, -0.19683117, -0.33119786, 0.19504476, 0.23377019]
        WeightsInit.randomize(biases.getValues()); //   [-0.07173675, -0.06975782, 0.46547735, -0.06449604, 0.22543085, -0.25612664, -0.16484225, -0.21565866, 0.45828927, -0.13396758]
        instance.setWeights(weights);
        instance.setBiases(biases.getValues());        
        instance.setDeltas(new Tensor(10));
        instance.deltaWeights = new Tensor(5, 10);
        WeightsInit.randomize(instance.deltaWeights.getValues()); // [-0.014675736, -0.20824462, -0.2544266, -0.29433697, 0.19522274, -0.042135, -0.2805665, 0.44587213, -0.38881636, 0.2882418, 0.06629634, 0.011203349, 0.014710844, 0.23828048, -0.34540945, -0.16317445, 0.052065372, 0.35782564, -0.2368347, 0.16372514, -0.21912044, 0.32790893, -0.08429372, -0.347282, -0.33204705, -0.25469518, 0.33288032, -0.22248, -0.19762617, -0.30643207, -0.05354631, -0.46328926, 0.23842907, 0.25631714, -0.4825107, 0.0952881, -0.32452244, -0.014563918, 0.48131067, -0.3766222, 0.44226605, -0.12013543, 0.35074377, -0.29712552, 0.259381, 0.49666756, -0.43016553, -0.40166676, -0.012900829, 0.3826325]
        WeightsInit.randomize(instance.deltaBiases); // [-0.32544816, -0.42210114, 0.46139675, 0.44991916, 0.11481196, 0.48068362, 0.2159949, 0.23206848, -0.41850775, 0.28163642]
                
        instance.applyWeightChanges();        
        Tensor expectedWeights = new Tensor(-0.30327034f, -0.23136496f, -0.1402145f, -0.65223693f, -0.27131938f, -0.10264724f, -0.09167038f,  0.0578903f,  -0.20470357f,  0.63714188f, -0.35844476f, -0.33212315f, 0.4615165f,   0.13870019f, -0.40853286f,  0.27610147f,  0.2458809f,   0.2741896f, -0.06043624f,  0.49228281f, -0.09033686f,  0.49088883f, -0.00958961f, -0.81646843f, -0.44407541f,  0.02067602f,  0.02200854f, -0.30582535f, -0.67090463f, -0.62103487f, -0.52198958f, -0.35070544f,  0.56883705f,  0.15350276f, -0.24242462f,  0.42086977f, 0.11694991f,  0.31048643f,  0.80725021f, -0.55393618f,  0.45647371f,  0.16600942f, 0.44481963f, -0.17407107f, -0.01234454f,  0.34930807f, -0.6269967f,  -0.73286462f, 0.18214393f,  0.61640269f);        
        Tensor expectedBiases = new Tensor(-0.39718491f, -0.49185896f, 0.9268741f, 0.38542312f, 0.34024281f, 0.22455698f, 0.05115265f, 0.01640982f, 0.03978152f, 0.14766884f);
                
        assertArrayEquals(weights.getValues(), expectedWeights.getValues(), 1e-7f);
        assertArrayEquals(biases.getValues(), expectedBiases.getValues(), 1e-7f);  
    }
    
}
