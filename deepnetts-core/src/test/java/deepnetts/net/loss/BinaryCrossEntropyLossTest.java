package deepnetts.net.loss;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.activation.ActivationType;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class BinaryCrossEntropyLossTest {
       
    /**
     * Test of addPatternError method, of class BinaryCrossEntropyLoss.
     */
    @Test
    public void testAddPatternError() {
        System.out.println("addPatternError");
        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                                        .addInputLayer(5)
                                        .addDenseLayer(10)
                                        .addOutputLayer(1, ActivationType.SIGMOID)
                                        .withLossFunction(LossType.CROSS_ENTROPY)
                                        .build();
        
        BinaryCrossEntropyLoss instance = (BinaryCrossEntropyLoss)neuralNet.getLossFunction();
       
        float[] actualOutput = new float[] {0.1f};        
        float[] targetOutput = new float[] {1.0f};        

        float[] result = instance.addPatternError(actualOutput, targetOutput);        
        float[] expResult = new float [] {-0.9f};
       
        assertArrayEquals(expResult, result, 1e-8f);
              
        // t * ln(y) + (1-t) * ln(1-y), where t is target and y actual
        // 1 * ln(0.1) + 0 * ln(0.9) = -2.30259
        
        float expTotalError = 2.30259f;
        float actualTotalError = instance.getTotalValue();
        
        assertEquals(expTotalError, actualTotalError, 1e-5f);
    }

    /**
     * Test of getTotalError method, of class BinaryCrossEntropyLoss.
     */
    @Test
    public void testGetTotalError() {
        System.out.println("getTotalError");
        
        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                                        .addInputLayer(5)
                                        .addDenseLayer(10)
                                        .addOutputLayer(1, ActivationType.SIGMOID)
                                        .withLossFunction(LossType.CROSS_ENTROPY)
                                        .build();
        
        BinaryCrossEntropyLoss instance = (BinaryCrossEntropyLoss)neuralNet.getLossFunction();
        
        float[] actualOutput = new float[] {0.1f};        
        float[] targetOutput = new float[] {1.0f};  

        float[] result = instance.addPatternError(actualOutput, targetOutput);        
        float[] expResult = new float [] {-0.9f};
       
        assertArrayEquals(expResult, result, 1e-8f);
        
        float expTotalError = 2.30259f; 
        float actualTotalError = instance.getTotalValue();
        
        assertEquals(expTotalError, actualTotalError, 1e-5f);        
                
        actualOutput = new float[] {0.7f};        
        targetOutput = new float[] {0.0f};  

        result = instance.addPatternError(actualOutput, targetOutput);        
        expResult = new float [] {0.7f};
       
        assertArrayEquals(expResult, result, 1e-8f);        
         
        // t * ln(y) + (1-t) * ln(1-y), where t is target and y actual
        // 0 * ln(0.7) + 1 * ln(0.3) = −1.203972804
        
        expTotalError = 1.75328f; // -0.5 * ( -2.30259 + -1.20397 )
        actualTotalError = instance.getTotalValue();
                      
        assertEquals(expTotalError, actualTotalError, 1e-5f);
    }

    /**
     * Test of reset method, of class BinaryCrossEntropyLoss.
     */
    @Test
    public void testReset() {
        System.out.println("reset");
        
        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                                        .addInputLayer(5)
                                        .addDenseLayer(10)
                                        .addOutputLayer(1, ActivationType.SIGMOID)
                                        .withLossFunction(LossType.CROSS_ENTROPY)
                                        .build();
        
        BinaryCrossEntropyLoss instance = (BinaryCrossEntropyLoss)neuralNet.getLossFunction();
       
        float[] actualOutput = new float[] {0.5f};        
        float[] targetOutput = new float[] {0.0f};

        float[] result = instance.addPatternError(actualOutput, targetOutput);        
        instance.reset();
        
        float actualTotalError = instance.getTotalValue();
        float expTotalError = Float.NaN;
        
        assertEquals(expTotalError, actualTotalError, 1e-8f);
        
    }
}
