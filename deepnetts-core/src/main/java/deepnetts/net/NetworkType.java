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

package deepnetts.net;

/**
 * Neural network architecture types
 * 
 * @author Zoran Sevarac
 */
public enum NetworkType {
    FEEDFORWARD("FEEDFORWARD"), CONVOLUTIONAL("CONVOLUTIONAL");
    
    private final String name;       

    private NetworkType(String s) {
        name = s;
    }    
    
    public boolean equalsName(String otherName) {
        return name.equals(otherName);
    }
    
    public static NetworkType Of(Class networkClass) {
        if (networkClass.equals(FeedForwardNetwork.class)) {
            return FEEDFORWARD;
        } else if (networkClass.equals(ConvolutionalNetwork.class)) {
            return CONVOLUTIONAL;
        }

       throw new RuntimeException("Unknown network type!");       
    }

    @Override
    public String toString() {
       return this.name;
    }        
}
