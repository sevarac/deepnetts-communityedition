package deepnetts.eval;

/**
 * this is an evaluation metric not a loss function
 * @author Zoran
 */
public class MeanSquaredError {
    
    private float totalSum;
    private int patternCount;
    
    public void add(float[] predicted, float[] target) {
        for(int i=0; i<predicted.length; i++)
            totalSum += Math.pow((predicted[i] - target[i]), 2);
        
        patternCount++;
    }
    
    
    public float getTotal() {
        return totalSum / patternCount;
    }
    
}