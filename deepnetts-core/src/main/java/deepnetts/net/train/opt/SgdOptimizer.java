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
import java.io.Serializable;

/**
 *
 * @author zoran
 */
public final class SgdOptimizer implements Optimizer, Serializable  {

	
	private static final long serialVersionUID = 8838615133822228400L;
	
	
    private float learningRate;
    // bias lr?
    
    public SgdOptimizer(AbstractLayer layer) {
        this.learningRate = layer.getLearningRate();
    }
    
    // index is not used!
    @Override
    public float calculateDeltaWeight(final float gradient, final int... index) {
        return -learningRate * gradient;
    }

    @Override
    public float calculateDeltaBias(float gradient, int idx) {
        return -learningRate * gradient;
    }

    
}
