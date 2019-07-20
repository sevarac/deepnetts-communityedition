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

package deepnetts.util;

/**
 * Static utility methods for tensors.
 * 
 * @author Zoran Sevarac
 */
public class Tensors {

    public static float[] copyOf(float[] arr) {
        float[] copy = new float[arr.length];
        System.arraycopy(arr, 0, copy, 0, arr.length);
        return copy;
    }

    public static void sub(float[] arr, float val) {
        for(int i=0; i < arr.length; i++) {
             arr[i] = arr[i] - val;
        }
    }

    public static void multiply(float[] arr1, float[] arr2) {
        for (int i = 0; i < arr1.length; i++) {
            arr1[i] *= arr2[i];
        }        
    }


    /**
     * Prevent instantiation of this class.
     */
    private Tensors() { }
    
    /**
     * Returns tensors with max value for each component of input tensors.
     * 
     * @param t 
     * @param max proposed max tensor
     * @return tensor with max value for each component
     */
    public static Tensor absMax(final Tensor t, final Tensor max) {
        final float[] tValues= t.getValues();
        final float[] maxValues= max.getValues();
        
        for(int i=0; i < tValues.length; i++) {
            if (Math.abs(tValues[i]) > maxValues[i]) maxValues[i] = Math.abs(tValues[i]);
        }
        return max;       
    }
    
    /**
     * Returns array with max values for each position in the given input vectors.
     * Stores max values in second parameter.
     * 
     * @param arr
     * @param max
     * @return 
     */
    public static float[] absMax(final float[] arr, final float[] max) {    
        for(int i=0; i < max.length; i++) {
            if (Math.abs(arr[i]) > max[i]) max[i] = Math.abs(arr[i]);
        }
        return max;       
    }    
    
    public static Tensor absMin(final Tensor t, final Tensor min) {
        final float[] tValues= t.getValues();
        final float[] minValues= min.getValues();
        
        for(int i=0; i < tValues.length; i++) {
            if (Math.abs(tValues[i]) < Math.abs(minValues[i])) minValues[i] = Math.abs(tValues[i]);
        }
        return min;       
    }    
    
    public static float[] absMin(final float[] arr, final float[] min) {       
        for(int i=0; i < arr.length; i++) {
            if (Math.abs(arr[i]) < Math.abs(min[i])) min[i] = Math.abs(arr[i]);
        }
        return min;       
    }        
    
    public static void div(final float[] array, final float val) {
        for (int i = 0; i < array.length; i++) {
            array[i] /= val;
        }
    }

    public static void div(final float[] array, final float[] divisor) {
        for (int i = 0; i < array.length; i++) {
            array[i] = array[i] / divisor[i];
        }
    }
    
    public static final void sub(final float[] array1, final float[] array2) {
        for (int i = 0; i < array1.length; i++) {
            array1[i] -= array2[i];
        }
    }

    public static final void add(final float[] array1, final float[] array2) {
        for (int i = 0; i < array1.length; i++) {
            array1[i] += array2[i];
        }
    }    
    
    // decimal scale
    
    // return min max mean std in one 
//    public static Tensor stats(Tensor t, Tensor min) {
//        final float[] tValues= t.getValues();
//        final float[] minValues= min.getValues();
//        
//        for(int i=0; i < tValues.length; i++) {
//            if (tValues[i] > minValues[i]) minValues[i] = tValues[i];
//        }
//        return min;       
//    }        
      
    public static Tensor zeros(int cols) {
        return new Tensor(cols, 0f);
    }
    
    public static Tensor ones(int cols) {
        return new Tensor(cols, 1.0f);
    }    
    
    public static Tensor random(int rows, int cols) {
        Tensor tensor = new Tensor(rows, cols);

        for (int r = 0; r < tensor.getRows(); r++) {
            for (int c = 0; c < tensor.getCols(); c++) {
                tensor.set(r, c, RandomGenerator.getDefault().nextFloat());
            }
        }
        return tensor;
    }
    
    public static Tensor random(int rows, int cols, int depth) {
        Tensor tensor = new Tensor(rows, cols, depth);

        for (int z = 0; z < tensor.getDepth(); z++) {
            for (int r = 0; r < tensor.getRows(); r++) {
                for (int c = 0; c < tensor.getCols(); c++) {
                    tensor.set(r, c, z, RandomGenerator.getDefault().nextFloat());
                }
            }
        }
        return tensor;
    }    
    
    public static Tensor random(int rows, int cols, int depth, int fourthDim) {
        Tensor tensor = new Tensor(rows, cols, depth, fourthDim);

        for (int f = 0; f < tensor.getFourthDim(); f++) {
            for (int z = 0; z < tensor.getDepth(); z++) {
                for (int r = 0; r < tensor.getRows(); r++) {
                    for (int c = 0; c < tensor.getCols(); c++) {
                        tensor.set(r, c, z, f, RandomGenerator.getDefault().nextFloat());
                    }
                }
            }
        }
        return tensor;
    }      
    
}
