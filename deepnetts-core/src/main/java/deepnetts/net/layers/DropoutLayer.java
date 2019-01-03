package deepnetts.net.layers;

/**
 *
 * @author zoran
 */
public class DropoutLayer extends AbstractLayer {

    @Override
    public void init() {
        // create tensor of same size as input from prev layer
    }

    @Override
    public void forward() {
        // randomize dropout filter with 1 and 0
        // weights.randomize();
        // direct multiply of input tensor
    }

    @Override
    public void backward() {
        // feed deltas backward usng same random dropout filter
    }

    @Override
    public void applyWeightChanges() {
        // this layer does nothing
    }
    
}
