package deepnetts.net.train;

import deepnetts.data.DataSet;

/**
 *
 * @author zoran
 */
public interface Trainable {
    
    public void train(DataSet<?> trainingSet);
    
}
