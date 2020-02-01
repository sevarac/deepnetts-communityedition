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
    
package deepnetts.data;

import deepnetts.util.Tensor;

/**
 * Single item in a Deep Netts data set that provides methods for accessing input and target output.
 * This could be a marker interface (turn to annotation in future)
 * for example @DataSetItem
 * If somebody wants data set if custm objectsthey have to implement this interface for those objects and it should work
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public interface ExampleDataItem {        
    
        public Tensor getInput();
        
        public Tensor getTargetOutput();
}
