package deepnetts.eval;

import java.util.HashMap;

/**
 *
 * @author Zoran
 */
public class PerformanceMeasure {
    // Regression metrics
    public final static String MEAN_ABSOLUTE_ERROR      = "MEAN_BSOLUTE_ERROR";
    public final static String MEAN_SQUARED_ERROR       = "MEAN_SQUARED_ERROR";
    public final static String ROOT_MEAN_SQUARED_ERROR  = "ROOT_MEAN_SQUARED_ERROR";
    public final static String R2                       = "R2";
    
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
