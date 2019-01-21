package deepnetts.net.layers;

import deepnetts.util.RandomGenerator;
import deepnetts.util.Tensor;

/**
 *
 * @author zoran
 */
public class DropoutLayer extends AbstractLayer {

    private Tensor dropout= new Tensor(); // binary tensor same dimensions as previous
    
    @Override
    public void init() {
        // create tensor of same size as input from prev layer
        // iza ovoga moze da bude fc layer ali i neki 2d ili 3d layer?
    }

    @Override
    public void forward() {
        // randomize dropout filter with 1 and 0
        // direct multiply of input tensor
        
        // bolje da  dropout ubacim u postojece layere kao podesavanje
        
        for(int i=0; i<dropout.size(); i++ ){
            float val = (RandomGenerator.getDefault().nextFloat() > 0.5?1f:0f);
            dropout.set(i, val);            
            outputs.set(i, val*inputs.get(i));            
        }                
    }

    @Override
    public void backward() {
        // feed deltas backward usng same random dropout filter
        for(int i=0; i<dropout.size(); i++ ){
            deltas.set(i,  gradients.get(i) * dropout.get(i));            
        }        
    }

    @Override
    public void applyWeightChanges() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
