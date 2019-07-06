package deepnetts.eval;

import java.util.HashMap;

/**
 * Wrapper for constants and values for classifier and regressor evaluation metrics.
 * 
 * @author Zoran Sevarac
 */
public class EvaluationMetrics {

    /**
    * Mean value of sum of squared errors.
    * Errors are squared in order to better explain variability, and take into account positive and negative errors.
    * Regression metrics
    */
    public final static String MEAN_ABSOLUTE_ERROR      = "MeanAbsoluteError";
    public final static String MEAN_SQUARED_ERROR       = "MeanSquaredError";
    public final static String ROOT_MEAN_SQUARED_ERROR  = "RootMeanSquaredError";
    public final static String RESIDUAL_SQUARE_SUM      = "ResidualSquareSum"; // RSS

    /**
     * Estimation of standard deviation of prediction errors for some given data set.
     * Smaller is better.
     */
    public final static String RESIDUAL_STANDARD_ERROR = "ResidualStandardError";

    /**
     * Percent of variation explained by the regression model.
     * 1 is good , 0 is bad, bigger is better.
     */
    public final static String R_SQUARED = "RSquared";

    /**
     * Is there a relationship between inputs and predictions(outputs) at all.
     * If there is a relationship, this value is greater then 1. 
     * When there is now relationship, this value is around 1.
     */
    public final static String F_STAT = "FStatistics";

    // p value? t statistics

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
