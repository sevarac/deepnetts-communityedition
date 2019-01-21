package deepnetts.net.layers;

import deepnetts.net.loss.LossType;
import deepnetts.net.train.opt.OptimizerType;
import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;
import deepnetts.util.WeightsInit;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Ignore;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class SoftmaxOutputLayerTest {
    
    /**
     * Test of forward method, of class SoftmaxOutputLayer.
     */
    @Test
    public void testForward() {
        System.out.println("forward");
        // initialize weights with specified random seed
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights2 (same effect as line above)
        
        // input vector for this layer
        Tensor input = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 3); // weights matrix   
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906]"
       
        // create prev fc layer with 5 outputs
        DenseLayer prevLayer = new DenseLayer(5);        
        prevLayer.setOutputs(input);
                
        // create instance of layer to test
        SoftmaxOutputLayer instance = new SoftmaxOutputLayer(3);
        instance.setPrevLayer(prevLayer);        
        instance.init(); // init weights structure
        instance.setWeights(weights); // set weights values
        instance.setBiases(new float[] {0.0f, 0.0f, 0.0f}); // set zero bias values

        // do the forward pass
        instance.forward();    
        
        // weighted sums      0.154060, -0.149083,   -0.034566 // octave 
        // calculated values  0.15406,  -0.14908276, -0.03456554
        // e^ws 1.166560878 0.861497814 0.966025024
        // sum(e^ws) = 2.994083716 
        // output = 0.389621997 0.287733375 0.322644627
                
        Tensor actualOutputs = instance.getOutputs();   // "[0.042127118, 0.36987683, 0.10604945, 0.24532129, 0.17567813, 0.34893453, 0.16589889, -0.3487752, 0.09166323, -0.015247092]"
        Tensor expectedOutputs = new Tensor( 0.389621997f, 0.287733375f, 0.322644627f);

        assertArrayEquals(actualOutputs.getValues(), expectedOutputs.getValues(), 1e-7f);
    }

    /**
     * Test of backward method, of class SoftmaxOutputLayer.
     */
    @Test
    public void testBackward() {
        System.out.println("backward");
        RandomGenerator.getDefault().initSeed(123); // init random generator with seed that will be used for weights (same effect as line above)
        
        // input vector for this layer
        Tensor inputs = new Tensor(0.1f, 0.2f, 0.3f, 0.4f, 0.5f);
        Tensor weights = new Tensor(5, 10); // weights from previous layer
        WeightsInit.uniform(weights.getValues(), 5); // "[0.19961303, -0.23501621, 0.43907326, -0.17747784, -0.22066136, 0.06630343, 0.097314, -0.21566293, 0.273578, 0.10945064, 0.33577937, 0.044093937, 0.19323963, -0.3021235, -0.38288906, 0.16261822, 0.26498383, -0.207817, 0.070406556, -0.23022851, 0.36503863, 0.091478825, -0.31402034, -0.25345784, 0.42504954, -0.037393004, -0.38854277, -0.36758634, -0.38503492, -0.33786723, -0.36604232, -0.14479709, -0.06755906, 0.38639867, 0.3348655, 0.15910655, 0.06717491, -0.4455302, -0.09257606, -1.219213E-4, -0.21616945, 0.43006968, -0.31055218, 0.2699433, -0.214278, 0.25471163, -0.03427276, -0.43431506, -0.054469943, -0.23747501]"
        Tensor outputErrors = new Tensor(10);
        outputErrors.setValues(0.04212712f, 0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);
                
        DenseLayer prevLayer = new DenseLayer(5); // not used for anything just dummy to prevent npe in init      
        prevLayer.setOutputs(inputs);
        
        SoftmaxOutputLayer instance = new SoftmaxOutputLayer(10);
        instance.setLossType(LossType.CROSS_ENTROPY);
        instance.setPrevLayer(prevLayer);
        instance.init();
        instance.setOptimizerType(OptimizerType.SGD);
        instance.setWeights(weights);
//        instance.setBiases(new float[] {0.1f, 0.2f, 0.3f, 0.11f, 0.12f, 0.13f, 0.21f, 0.22f, 0.23f, 0.24f}); // set bias values
//        instance.forward(); // derivatives are calculated using outputs | outputs : "[0.51053023, 0.59142923, 0.5264875, 0.5610246, 0.5438069, 0.5863592, 0.5413798, 0.41367948, 0.52289975, 0.4961883]"          
        instance.setOutputErrors(outputErrors.getValues());

        instance.backward();
                
        Tensor result = instance.getDeltas(); //   [0.01052711, 0.08937729, 0.026437959, 0.060416743, 0.043582395, 0.08463131, 0.04119066, -0.084595, 0.022867741, -0.0038115513]
        Tensor expResult = new Tensor(10); // [0.01052711  0.08937729  0.02643796   0.06041675   0.0435824    0.08463131  0.04119066  -0.084595  0.02286774   -0.00381155] from num py test
        expResult.setValues( 0.04212712f,  0.3698768f, 0.10604945f, 0.24532129f, 0.17567812f, 0.34893453f, 0.16589892f, -0.34877524f, 0.09166324f, -0.01524709f);

        assertArrayEquals(expResult.getValues(), result.getValues(), 1e-8f);        
        
        Tensor deltaWeights = instance.getDeltaWeights();
        Tensor expDeltaWeights = new Tensor(-0.000421271f,  -0.00369877f,  -0.00106049f,  -0.00245321f,  -0.00175678f,  -0.00348935f,  -0.00165899f,  0.00348775f,  -0.000916632f,  0.000152471f,  -0.000842543f,  -0.00739754f,  -0.00212099f,  -0.00490643f,  -0.00351356f,  -0.00697869f,  -0.00331798f,  0.00697551f,  -0.00183326f,  0.000304942f,  -0.00126381f,  -0.0110963f,  -0.00318148f,  -0.00735964f,  -0.00527034f,  -0.010468f,  -0.00497697f,  0.0104633f,  -0.0027499f,  0.000457413f,  -0.00168509f,  -0.0147951f,  -0.00424198f,  -0.00981285f,  -0.00702712f,  -0.0139574f,  -0.00663596f,  0.013951f,  -0.00366653f,  0.000609884f,  -0.00210636f,  -0.0184938f,  -0.00530247f,  -0.0122661f,  -0.00878391f,  -0.0174467f,  -0.00829495f,  0.0174388f,  -0.00458316f,  0.000762355f);
                
        assertArrayEquals(expDeltaWeights.getValues(), deltaWeights.getValues(), 1e-7f);
    }
    
}