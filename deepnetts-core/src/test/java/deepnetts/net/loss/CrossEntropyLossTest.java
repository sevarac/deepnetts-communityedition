package deepnetts.net.loss;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.ActivationType;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Ignore;

/**
 * KOpiraj delove binary ce i po uzoru na intermezzo i nndl
 * testiray u pythonu
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class CrossEntropyLossTest {
    
    /**
     * Test of addPatternError method, of class CrossEntropyLoss.
     */
    @Test
    public void testAddPatternError() {
        System.out.println("addPatternError");
        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                                        .addInputLayer(5)
                                        .addDenseLayer(10)
                                        .addOutputLayer(3, ActivationType.SOFTMAX)
                                        .withLossFunction(LossType.CROSS_ENTROPY)
                                        .build();
        
        CrossEntropyLoss instance = (CrossEntropyLoss)neuralNet.getLossFunction();
              
        float[] actualOutput = new float[] {0.1f, 0.2f, 0.7f};        
        float[] targetOutput = new float[] {0.0f, 0.0f, 1.0f};        

        float[] result = instance.addPatternError(actualOutput, targetOutput);        
        float[] expResult = new float [] {0.1f, 0.2f, -0.3f};
       
        assertArrayEquals(expResult, result, 1e-8f);
              
        //  ln(0.7) = −0.356674944
        // -sum(ln(actualTargetY)) =  0.356674944
        
        float expTotalError = 0.356674944f;
        float actualTotalError = instance.getTotalValue();
        
        assertEquals(expTotalError, actualTotalError, 1e-7f);
    }

    /**
     * Test of getTotalError method, of class CrossEntropyLoss.
     */
    @Test
    public void testGetTotalError() {
        System.out.println("getTotalError");
        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                                        .addInputLayer(5)
                                        .addDenseLayer(10)
                                        .addOutputLayer(3, ActivationType.SOFTMAX)
                                        .withLossFunction(LossType.CROSS_ENTROPY)
                                        .build();
        
        CrossEntropyLoss instance = (CrossEntropyLoss)neuralNet.getLossFunction();
              
        float[] actualOutput = new float[] {0.1f, 0.2f, 0.7f};        
        float[] targetOutput = new float[] {0.0f, 0.0f, 1.0f};        

        float[] result = instance.addPatternError(actualOutput, targetOutput);        
        float[] expResult = new float [] {0.1f, 0.2f, -0.3f};
       
        assertArrayEquals(expResult, result, 1e-8f);
                      
        //  ln(0.7) = −0.356674944
        // -sum(ln(actualTargetY)) =  0.356674944        
        
        float expTotalError = 0.356674944f;
        float actualTotalError = instance.getTotalValue();
        
        assertEquals(expTotalError, actualTotalError, 1e-7f);
        
        actualOutput = new float[] {0.2f, 0.8f, 0.3f};       
        targetOutput = new float[] {0.0f, 1.0f, 0.0f};   

        result = instance.addPatternError(actualOutput, targetOutput);        
        expResult = new float [] {0.2f, -0.2f, 0.3f};
       
        assertArrayEquals(expResult, result, 1e-7f);        
         
        // ln(0.8) = −0.223143551      
        // -sum(ln(actualTargetY))
        // −0.356674944 + −0.223143551 = −0.579818495
        // total error = -sum/2
        
        expTotalError = 0.289909248f; // -0.5 * ( -2.30259 + -1.20397 )
        actualTotalError = instance.getTotalValue();
                      
        assertEquals(expTotalError, actualTotalError, 1e-5f);        
        
    }

    /**
     * Test of reset method, of class CrossEntropyLoss.
     */
    @Test
    public void testReset() {
        System.out.println("reset");
        
        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                                        .addInputLayer(5)
                                        .addDenseLayer(10)
                                        .addOutputLayer(3, ActivationType.SOFTMAX)
                                        .withLossFunction(LossType.CROSS_ENTROPY)
                                        .build();
        
        CrossEntropyLoss instance = (CrossEntropyLoss)neuralNet.getLossFunction();
       
        float[] actualOutput = new float[] {0.1f, 0.2f, 0.7f};        
        float[] targetOutput = new float[] {0.0f, 0.0f, 1.0f};  

        float[] result = instance.addPatternError(actualOutput, targetOutput);        
        instance.reset();
        
        float actualTotalError = instance.getTotalValue();
        float expTotalError = Float.NaN;
        
        assertEquals(expTotalError, actualTotalError, 1e-8f);
    }
    
}
