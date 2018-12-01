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
    
package deepnetts.net.loss;

/**
 * Types of loss functions
 * 
 * @author zoran
 */
public enum LossType {
    MEAN_SQUARED_ERROR("MEAN_SQUARED_ERROR"), CROSS_ENTROPY("CROSS_ENTROPY");
    
    private final String name;       

    private LossType(String name) {
        this.name = name;
    }    
    
    public boolean equalsName(String otherName) {
        return name.equals(otherName);
    }

    public static LossType of(Class lossClass) {
        if (lossClass.equals(MeanSquaredErrorLoss.class)) {
            return MEAN_SQUARED_ERROR;
        } else if (lossClass.equals(CrossEntropyLoss.class) || lossClass.equals(BinaryCrossEntropyLoss.class)) {
            return CROSS_ENTROPY;
        }

       throw new RuntimeException("Unknown loss type!");       
    }    
    
    @Override
    public String toString() {
       return this.name;
    }      
}