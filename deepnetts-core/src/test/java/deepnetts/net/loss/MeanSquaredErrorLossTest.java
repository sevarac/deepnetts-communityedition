package deepnetts.net.loss;

import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.ActivationType;
import deepnetts.util.RandomGenerator;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class MeanSquaredErrorLossTest {
    

    /**
     * Test of addPatternError method, of class MeanSquaredErrorLoss.
     * Test for a single error vector: calculation of error vector and total error
     */
    @Test
    public void testAddPatternError() {
        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                                        .addInputLayer(5)
                                        .addDenseLayer(10)
                                        .addOutputLayer(3, ActivationType.LINEAR)
                                        .withLossFunction(LossType.MEAN_SQUARED_ERROR)
                                        .build();
        
        MeanSquaredErrorLoss instance = (MeanSquaredErrorLoss)neuralNet.getLossFunction();
       
        float[] actualOutput = new float[] {0.5f, 0.4f, 0.3f};        
        float[] targetOutput = new float[] {0.12f, 0.2f, 0.4f};

        float[] result = instance.addPatternError(actualOutput, targetOutput);        
        float[] expResult = new float [] {0.38f, 0.2f, -0.1f};
       
        assertArrayEquals(expResult, result, 1e-8f);
        
        float expTotalError = 0.0972f; // (0.38*0.38 + 0.2*0.2 + −0.1 * −0.1) / 2
        float actualTotalError = instance.getTotalValue();
        
        assertEquals(expTotalError, actualTotalError, 1e-8f);
        
    }

    /**
     * Test of getTotalError method, of class MeanSquaredErrorLoss.
     */
    @Test
    public void testGetTotalError() {
        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                                        .addInputLayer(5)
                                        .addDenseLayer(10)
                                        .addOutputLayer(3, ActivationType.LINEAR)
                                        .withLossFunction(LossType.MEAN_SQUARED_ERROR)
                                        .build();
        
        MeanSquaredErrorLoss instance = (MeanSquaredErrorLoss)neuralNet.getLossFunction();
       
        float[] actualOutput = new float[] {0.5f, 0.4f, 0.3f};        
        float[] targetOutput = new float[] {0.12f, 0.2f, 0.4f};

        float[] result = instance.addPatternError(actualOutput, targetOutput);        
        float[] expResult = new float [] {0.38f, 0.2f, -0.1f};
       
        assertArrayEquals(expResult, result, 1e-8f);

        float expTotalError = 0.0972f; // (0.38*0.38 + 0.2*0.2 + −0.1 * −0.1) / 2
        float actualTotalError = instance.getTotalValue();
        
        assertEquals(expTotalError, actualTotalError, 1e-8f);
        
        actualOutput = new float[] {0.1f, 0.3f, 0.7f};        
        targetOutput = new float[] {0.2f, 0.21f, 0.32f};

        result = instance.addPatternError(actualOutput, targetOutput);        
        expResult = new float [] {-0.1f, 0.09f, 0.38f};
        
        assertArrayEquals(expResult, result, 1e-7f);
        
        expTotalError = 0.089225f; // ( 0.0972 +  (-0.1*-0.1 + 0.09*0.09 + 0.38*0.38)/2 ) /2
        actualTotalError = instance.getTotalValue();
        
        assertEquals(expTotalError, actualTotalError, 1e-8f);
    }

    /**
     * Test of reset method, of class MeanSquaredErrorLoss.
     */
    @Test
    public void testReset() {

        NeuralNetwork neuralNet = FeedForwardNetwork.builder()
                                        .addInputLayer(5)
                                        .addDenseLayer(10)
                                        .addOutputLayer(3, ActivationType.LINEAR)
                                        .withLossFunction(LossType.MEAN_SQUARED_ERROR)
                                        .build();
        
        MeanSquaredErrorLoss instance = (MeanSquaredErrorLoss)neuralNet.getLossFunction();
       
        float[] actualOutput = new float[] {0.5f, 0.4f, 0.3f};        
        float[] targetOutput = new float[] {0.12f, 0.2f, 0.4f};

        float[] result = instance.addPatternError(actualOutput, targetOutput);        
        instance.reset();
        
        float actualTotalError = instance.getTotalValue();
        float expTotalError = Float.NaN;
        
        assertEquals(expTotalError, actualTotalError, 1e-8f);
        
    }
    
}
