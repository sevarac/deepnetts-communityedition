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

import java.util.HashMap;

/**
 *
 * @author Zoran
 */
public class PerformanceMeasure {
    public final static String MSE          = "MSE";
    public final static String ACCURACY     = "Accuracy";
    public final static String PRECISION    = "Precision";
    public final static String RECALL       = "Recall";
    public final static String F1SCORE      = "F1Score";
    
    private final HashMap<String, Float> values = new HashMap();
    
    public float get(String key) {
        return values.get(key);
    }
    
    public void set(String key, float value) {
        values.put(key, value);
    }
    
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        values.entrySet().stream().forEach((e) ->  sb.append(e.getKey() + ": "+e.getValue() + System.lineSeparator()) );
                
        return sb.toString();
    }
    
}
