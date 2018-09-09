package deepnetts.net.layers;

import deepnetts.util.Tensor;

public interface Layer {

    public void forward();
    public void backward();

    public Tensor getOutputs();

//    public Tensor getDeltas();
//    public Tensor getWeights();

}
