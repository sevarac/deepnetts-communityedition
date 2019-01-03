package deepnetts.net.train;

import deepnetts.data.DataSet;

/**
 *
 * @author zoran
 */
public interface Trainer {
        public void train(DataSet<?> trainingSet);
}
