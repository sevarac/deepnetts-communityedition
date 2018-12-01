/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deepnetts.net.train.opt;

import deepnetts.net.layers.AbstractLayer;
import deepnetts.util.Tensor;

/**
 *
 * @author Zoran
 */
@FunctionalInterface
public interface Optimizer {

    public void optimize(AbstractLayer layer);
    //public void optimize(Tensor outputs, Tensor weights, Tensor grads );
}
