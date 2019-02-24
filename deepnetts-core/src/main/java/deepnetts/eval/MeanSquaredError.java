package deepnetts.eval;

/**
 * This class calculates values used for evaluation metrics for regression problems.
 *
 * @author Zoran Sevarac
 */
public class MeanSquaredError {

    private float squaredSum;
    private int patternCount;

    public void add(float[] predicted, float[] target) {
        for(int i=0; i<predicted.length; i++)
            squaredSum += Math.pow((predicted[i] - target[i]), 2);

        patternCount++;
    }

    /**
     * Returns squared error sum (RSS, or residual square sum)
     * @return
     */
    public float getSquaredSum() {
        return squaredSum;
    }

    /**
     * Returns mean squared error
     * @return
     */
    public float getMeanSquaredSum() {
        return squaredSum / patternCount;
    }

    public float getRootMeanSquaredSum() {
        return (float)Math.sqrt(squaredSum / patternCount);
    }

}