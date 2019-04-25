/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */

package deepnetts.net.weights;

import deepnetts.util.RandomGenerator;

/**
 * This class provides various randomization methods.
 * 
 * @author Zoran Sevarac
 */
public class RandomWeights {
    
    private static RandomGenerator randomGen =  RandomGenerator.getDefault();
    
    public static void initSeed(long seed) {
        RandomGenerator.getDefault().initSeed(seed);
    }
        
    /**
     * Fills the specified array with random numbers in range [-0.5, 0.5] from the current random seed
     * @param array 
     */
    public static void randomize(float[] array) {
        for (int i = 0; i < array.length; i++) {
           array[i] = randomGen.nextFloat() - 0.5f;
        }
    }
  
        
    public static void widrowHoff(float[] array, float input, float hidden) {         
        randomize(array);
        float beta = 0.7f * (float)Math.pow(hidden, 1/input);   //1/input* 
        
        float weightsNorm =0;
        for (int i = 0; i < array.length; i++) {
            weightsNorm += array[i]*array[i];
        }
        weightsNorm = (float)Math.sqrt(weightsNorm);
        
        for (int i = 0; i < array.length; i++) {
            array[i] = (beta*array[i]) / weightsNorm;
        }
    }    
    
    
    /**
     *  Uniform U[-a,a] where a=1/sqrt(in). 
     *   
     * Understanding the difficulty of training deep feedforward neural networks
     * Xavier Glorot, Yoshua Bengio, 2010, Commonly used "heuristic" , Eq. 1 pg 251
     * http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
     * 
     * https://en.wikipedia.org/wiki/Uniform_distribution_(continuous)
     * 
     *  ovo je kao xavier sa 1 u brojiocu     
     * @param weights an array of weights 
     * @param numInputs number of inputs, a size of the previous layer
     */
    public static void uniform(float[] weights, int numInputs) {        
        if (numInputs==0) throw new IllegalArgumentException("Number of inputs for uniform randomization cannot be zero!");
        
        float min = -1 / (float)Math.sqrt(numInputs);
        float max = 1 / (float)Math.sqrt(numInputs);
      
        for (int i = 0; i < weights.length; i++) {
           weights[i] =  min + (randomGen.nextFloat()* (max-min));
        }        
    }
    
    public static void uniform(float[] weights, float min, float max) {              
       for (int i = 0; i < weights.length; i++) {
           weights[i] =  min + (randomGen.nextFloat()* (max-min));
       }             
    }
    
    // use for relus
    // Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    // Koristi za Relu i leaky relu
    public static void he(float[] weights, int numInputs) {      
        gaussian(weights, 0, (float)Math.sqrt(2.0f/numInputs));
    }    
    
    // Box–Muller transform
    // https://www.javamex.com/tutorials/random_numbers/gaussian_distribution_2.shtml
    // http://mathworld.wolfram.com/Box-MullerTransformation.html
    // https://stackoverflow.com/questions/19944111/creating-a-gaussian-random-generator-with-a-mean-and-standard-deviation    
      // 70% of values will be between -1 and 1
    public static void gaussian(float[] weights, float mean, float std) {      
       for (int i = 0; i < weights.length; i++) {
           weights[i] =  randomGen.nextGaussian()*std + mean;       
       }
    }   
    
    // 70% of values will be between -1 and 1
    // zero mean, 1 std
    public static void normal(float[] weights) {      
       gaussian(weights, 0, 1); 
    }     
    
    /**
     * Normalized uniform initialization U[-a,a] with a = sqrt(6/(in + out)).
     *  
     * Xavier Glorot, Yoshua Bengio, 2010, 
     * Understanding the difficulty of training deep feedforward neural networks
     * http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf  ,  pg. 253, eq 16.
     * 
     * Use for tanh 
     * 
     * @param weights
     * @param numIn  size of the previous layer (number of inputs)
     * @param numOut size of initialized layer (number of outputs)
     * 
     * https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78
     */
    public static void xavier(float[] weights, int numIn, int numOut) {
        float min = (float)-Math.sqrt( 6 / (float)(numIn+numOut));
        float max = (float)Math.sqrt( 6 / (float)(numIn+numOut));

        for (int i = 0; i < weights.length; i++) {
           weights[i] =  min + (randomGen.nextFloat() * (max-min));
        }                       
    }
}
