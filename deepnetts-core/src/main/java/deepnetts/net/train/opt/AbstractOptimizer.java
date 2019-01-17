package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.util.Tensor;

// ovaj da moze da radi na output i dense layer
public abstract class AbstractOptimizer  {

    protected AbstractLayer layer;
    
    protected Tensor gradients; // gradients are salculated in layer    
    protected Tensor deltas;
    protected Tensor inputs;
    
    // uloga optimizaera je da uzracuna delta weight 

    protected float learningRate;   // this can also be for each weight!
   
    protected Tensor deltaWeights;
    protected float[] deltaBiases;

    public AbstractOptimizer(AbstractLayer layer) {
        this.layer = layer;
        gradients = layer.getGradients();   // sve ove takodje inicijalizuj u konstruktoru
        deltas = layer.getDeltas();
        deltaWeights = layer.getDeltaWeights();
        deltaBiases = layer.getBiases();
        inputs = layer.getPrevlayer().getOutputs();
        learningRate = layer.getLearningRate();        
    }
    
    // ovo treba napraviti tako da na istu osnovu mogu lako da se dodaju optimizeri
    // kako da radi i za 2d i 3d layere
    // ovo je prakticno step 2 iz backward-a
  //  @Override
    public void optimize() { // layer treba proslediti konstruktru optimizera
        // iteriraj neurone/delte i ulaze i za svaki ulaz izracunaj promenu tezine
        for (int deltaCol = 0; deltaCol < deltas.getCols(); deltaCol++) {   // this iterates neurons/deltas
            for (int inCol = 0; inCol < inputs.getCols(); inCol++) {        // iterate inputs for each neuron
                final float grad = deltas.get(deltaCol) * inputs.get(inCol); // calculate gradient dE/dw = delta * input_of_the_realated_weight
                gradients.set(inCol, deltaCol, grad);
                
                final float deltaWeight = calculate(grad);
                deltaWeights.add(inCol, deltaCol, deltaWeight); // accumulate delta weighsts    // add or set?
            }

            final float deltaBias = calculate(deltas.get(deltaCol));
            deltaBiases[deltaCol] += deltaBias;
        }
    }

    // this method implemenst specific formulas in subclasses
    public abstract float calculate(final float grad);

}
