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

import java.awt.Color;

/**
 *
 * @author Zoran Sevarac
 */
public class ColorUtils {

    private ColorUtils() { }

    public final static int getRed(final int color) {
        return (color >> 16) & 0xFF;
    }

    public final static int getGreen(final int color) {
        return (color >> 8) & 0xFF;
    }

    public final static int getBlue(final int color) {
        return color & 0xFF;
    }
    
    public static Color getColorFor(float min, float max, float val) { // value [0..1]
        int r, g, b;        
        float ratio = 2 * (val-min) / ((float)(max-min));
        
        b = (int)Math.max(0, 255*(1-ratio));
        r = (int)Math.max(0, 255*(ratio-1));
        g = 255 - b - r;
        System.out.println(val);
        Color color = new Color(r, g, b, 255);
        return color;
    }
  
}
