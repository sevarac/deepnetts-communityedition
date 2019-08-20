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

package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.util.DeepNettsException;

public interface Optimizer {
    
    public float calculateDeltaWeight(final float gradient, final int... index);
    public float calculateDeltaBias(final float gradient, final int idx);
         
    public static Optimizer create(OptimizerType type, AbstractLayer layer) {
        switch (type) {
            case SGD:
                return new SgdOptimizer(layer);
            case MOMENTUM:
                return new MomentumOptimizer(layer);
            default:
                throw new DeepNettsException("Unknown optimizer:" + type);
        }
    }    
    
}
