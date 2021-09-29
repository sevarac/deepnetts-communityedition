package deepnetts.util;


import java.io.ObjectStreamClass;


import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.net.layers.ConvolutionalLayer;
import deepnetts.net.layers.FullyConnectedLayer;
import deepnetts.net.layers.InputLayer;
import deepnetts.net.layers.MaxPoolingLayer;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.layers.SoftmaxOutputLayer;
import deepnetts.net.loss.CrossEntropyLoss;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.opt.SgdOptimizer;


public class Serialization {
	
	
	public static void main(String[] args) {
		
		Class[] dataModel = new Class[] {NeuralNetwork.class, ConvolutionalNetwork.class, BackpropagationTrainer.class,
				AbstractLayer.class, InputLayer.class, OutputLayer.class, ConvolutionalLayer.class, FullyConnectedLayer.class,
				MaxPoolingLayer.class, SoftmaxOutputLayer.class,
				Tensor.class,
				SgdOptimizer.class,
				CrossEntropyLoss.class};
		
		
		for (int i = 0; i < dataModel.length; i++) {
			System.out.println("class=" + dataModel[i] + ", serialVersionID=" + ObjectStreamClass.lookup(dataModel[i]).getSerialVersionUID());
		}
	}
}
