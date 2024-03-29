/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */

package deepnetts.eval;

/**
 * this is an evaluation metric not a loss function
 * @author Zoran
 */
public class MeanSquaredError {
    
    private float totalSum;
    private int patternCount;
    
    public void add(float[] predicted, float[] target) {
        for(int i=0; i<predicted.length; i++)
            totalSum += Math.pow((predicted[i] - target[i]), 2);
        
        patternCount++;
    }
    
    
    public float getTotal() {
        return totalSum / patternCount;
    }
    
}