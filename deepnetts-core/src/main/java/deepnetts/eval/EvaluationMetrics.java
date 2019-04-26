package deepnetts.eval;

import java.util.HashMap;

/**
 *
 * @author Zoran
 */
public class EvaluationMetrics {

    /**
    * Mean value of sum of squared errors.
    * Errors are squared in order to better explain variability, and take into account positive and negative errors.
    * Regression metrics
    */
    public final static String MEAN_ABSOLUTE_ERROR      = "MeanAbsoluteError";
    public final static String MEAN_SQUARED_ERROR       = "MeanSquaredError";
    public final static String ROOT_MEAN_SQUARED_ERROR  = "RootMeanSquaredError";   // Use RSE instead
    public final static String RESIDUAL_SQUARE_SUM      = "ResidualSquareSum";

    public final static String RESIDUAL_STANDARD_ERROR  = "ResidualStandardError"; // Smaller is better. Average error of estimated/predicted outputs of the regression model. (or standard deviation of errors)
    public final static String R2                       = "RSquared"; //  Bigger is better. Percent of variation explained by the regression model
    public final static String F_STAT                   = "FStatistics";

    // Classification Metrics
    public final static String ACCURACY     = "Accuracy";
    public final static String PRECISION    = "Precision";
    public final static String RECALL       = "Recall";
    public final static String F1SCORE      = "F1Score";


    private final HashMap<String, Float> values = new HashMap();

    public float get(String key) {
        return values.get(key);
    }

    public void set(String key, float value) {
        values.put(key, value);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        values.entrySet().stream().forEach((e) ->  sb.append(e.getKey() + ": "+e.getValue() + System.lineSeparator()) );

        return sb.toString();
    }

}
