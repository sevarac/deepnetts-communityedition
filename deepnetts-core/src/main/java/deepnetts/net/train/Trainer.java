package deepnetts.net.train;

import deepnetts.data.DataSet;


public interface Trainer {
        public void train(DataSet<?> trainingSet);
     //   public void train(DataSet<?> trainingSet, DataSet<?> validationSet);
    //    public void train(DataSet<?> trainingSet, double validationRatio);
}
