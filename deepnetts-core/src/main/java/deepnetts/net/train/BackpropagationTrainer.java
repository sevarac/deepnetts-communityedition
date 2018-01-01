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
    
package deepnetts.net.train;

import deepnetts.core.DeepNetts;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.loss.LossFunction;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.apache.logging.log4j.LogManager;

/**
 * This class implements training algorithm for feed forward and convolutional neural network.
 * 
 * @author Zoran Sevarac <zoran.sevarac@smart4net.co>
 */
public class BackpropagationTrainer implements Serializable {

    /**
     * Maximum training epochs.
     * Training will stop when this number of epochs is reached regardless the total network error.
     */
    private long maxEpochs = 100000L;
    
    /**
     * Maximum allowed error.
     * Training will stop once total error has reached this value .
     */
    private float maxError = 0.01f;
    
    /**
     * Learning rate.
     */
    float learningRate = 0.01f;
    
    /**
     * Optimizer type for all layers
     */
    private OptimizerType optimizer = OptimizerType.SGD;
    
    private float momentum = 0;
   
    private boolean batchMode = false;
    private int batchSize;    
    private boolean stopTraining = false;
    
    private int epoch;
    private float totalError;
      
    private LossFunction lossFunction;
    
    private List<TrainingListener> listeners = new ArrayList<>();
    
    public static final String PROP_MAX_ERROR = "maxError";
    public static final String PROP_MAX_ITERATIONS = "maxIterations";
    public static final String PROP_LEARNING_RATE = "learningRate";
    public static final String PROP_MOMENTUM = "momentum";
    public static final String PROP_BATCH_MODE = "batchMode";
    
    /**
     * Neural network to train
     */
    private NeuralNetwork neuralNet;

    private static final org.apache.logging.log4j.Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());    
    

    public BackpropagationTrainer(NeuralNetwork neuralNet) { 
        
        if (neuralNet == null) throw new IllegalArgumentException("Parameter neuralNet cannot be null!");
        
        this.neuralNet = neuralNet;
    }
    
    public BackpropagationTrainer(NeuralNetwork neuralNet, Properties prop) { 
        this(neuralNet);
                
        this.maxError = Float.parseFloat(prop.getProperty(PROP_MAX_ERROR));
        this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_ITERATIONS));
        this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));
        this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
        this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));       
        // TODO: set optimizer type
        
    }    

    
    /**
     * This method does actual training procedure.
     * 
     * Make this pure function so it can run in multithreaded - can train several nn in parallel
     * put network as param
     * 
     * @param dataSet 
     */    
    public void train(DataSet<?> dataSet) {

        if (dataSet == null) throw new IllegalArgumentException("Argument dataSet cannot be null!");
        if (dataSet.size() == 0) throw new RuntimeException("Data set is empty!");
        
        // neuralNet.setOutputLabels(dataSet.getLabels());
        //LOGGER.info(FileIO.toJson((ConvolutionalNetwork)neuralNet));
                
        int trainingSamplesCount = dataSet.size();   
        stopTraining = false;
     
        if (batchMode && (batchSize==0)) batchSize = trainingSamplesCount; 
        
        // set same lr to all layers!
        for(AbstractLayer layer : neuralNet.getLayers()) {
            layer.setLearningRate(learningRate);
            layer.setMomentum(momentum);
            layer.setBatchMode(batchMode);
            layer.setBatchSize(batchSize);
            layer.setOptimizer(optimizer);
        }
     
        lossFunction = neuralNet.getLossFunction();
                
        float[] outputError; 
        epoch = 0;                
        totalError = 0;
        float prevTotalError=0, totalErrorChange;
        long startTraining, endTraining, trainingTime, startEpoch, endEpoch, epochTime;
        
        fireTrainingEvent(TrainingEvent.Type.STARTED);    

        startTraining = System.currentTimeMillis();
        do {
            epoch++;
//            totalError = 0; 
            lossFunction.reset();
            int sampleCounter = 0;
            
            startEpoch = System.currentTimeMillis();
                      
            for (DataSetItem dataSetItem : dataSet) {       // for all items in dataset 
                sampleCounter++;
                neuralNet.setInput(dataSetItem.getInput());   // set network input     
                neuralNet.forward();                                // do forward pass / calculate network output                         
                
                outputError = lossFunction.addPatternError(neuralNet.getOutput(), dataSetItem.getTargetOutput()); // get output error using loss function
                neuralNet.setOutputError(outputError); //mozda bi ovo moglao da bude uvek isti niz/reference pa ne mora da se seuje                
                //totalError += lossFunction.getPatternError();  // maybe sum this in loass function                              
                
                neuralNet.backward();                                 // do the backward propagation using current outputError - should I use outputError as a param here?
                             
//                if (LOGGER.getLevel().intValue() <= Level.FINE.intValue()) {
//                    LOGGER.log(Level.INFO, ConvNetLogger.getInstance().logNetwork(neuralNet));  //  log the network details (outputs, wiights deltas ... )- for debugging purposes
//                }    
                
                // weight update for online mode after each training pattern
                if (!isBatchMode()) { // for online training update weight changes after each pass
                    neuralNet.applyWeightChanges();   
                } 
                else if (sampleCounter % batchSize == 0) { // mini batch
                    //LOG.info("Weight Update after: "+sampleCounter);    // ovde logovati mini batch, mozda i bacati event
                    neuralNet.applyWeightChanges();   
                }
                
                fireTrainingEvent(TrainingEvent.Type.ITERATION_FINISHED);  // move this inside for loop so we can track each pattern                           
                
            }
            endEpoch = System.currentTimeMillis(); 
            
          //   batch weight update after entire data set - ako vrlicina dataseta nije deljiva sa batchSize - ostatak
            if (isBatchMode() && (trainingSamplesCount % batchSize !=0 )) { // full batch. zarga ovaj gore ne pokriva?
                neuralNet.applyWeightChanges();
            }           
                                    
            // totalError = totalError / trainingSamplesCount; // - da li total error za ceo data set ili samo za mini  batch? lossFunction.getTotalError()
            totalError = lossFunction.getTotalError(); // - da li total error za ceo data set ili samo za mini  batch? lossFunction.getTotalError()
            
            totalErrorChange = totalError - prevTotalError; // todo: pamti istoriju ovoga i crtaj funkciju, to je brzina konvergencije na 10, 100, 1000 iteracija paterna - ovo treba meriti. Ovo moze i u loss funkciji
            prevTotalError = totalError;
            epochTime = endEpoch-startEpoch;

            LOGGER.info("Epoch:" + epoch + ", Time:"+epochTime + "ms, TotalError:" + totalError +", ErrorChange:"+totalErrorChange); // EpochTime:"+epochTime + "ms,
    //        LOG.log(Level.INFO, ConvNetLogger.getInstance().logNetwork(neuralNet));  
            
            fireTrainingEvent(TrainingEvent.Type.EPOCH_FINISHED);
            
            stopTraining = ((epoch == maxEpochs) || (totalError <= maxError)) || stopTraining;
            
        } while (!stopTraining); // or learning slowed, or overfitting, ...    
        
        endTraining = System.currentTimeMillis();
        trainingTime = endTraining - startTraining;
        
        LOGGER.info("Total Training Time: " + trainingTime + "ms");
        
        fireTrainingEvent(TrainingEvent.Type.STOPPED);
    }

    public long getMaxEpochs() {
        return maxEpochs;
    }

    public BackpropagationTrainer setMaxEpochs(long maxEpochs) {
        this.maxEpochs = maxEpochs;
        return this;
    }

    public float getMaxError() {
        return maxError;
    }

    // cannot be negative
    public BackpropagationTrainer setMaxError(float maxError) {
        this.maxError = maxError;
        return this;
    }

    // cannot be negative
    public BackpropagationTrainer setLearningRate(float learningRate) {
        this.learningRate = learningRate;
        return this;
    }
    
   
    private void fireTrainingEvent(TrainingEvent.Type type) {
        for(TrainingListener l : listeners) {
            l.handleEvent(new TrainingEvent<>(this, type));
        }
    }
    
    public void addListener(TrainingListener listener) {
        if (!listeners.contains(listener)) {
            listeners.add(listener);
        }
    }

    public void removeListener(TrainingListener listener) {
        listeners.remove(listener);
    }

    public boolean isBatchMode() {
        return batchMode;
    }

    public BackpropagationTrainer setBatchMode(boolean batchMode) {
        this.batchMode = batchMode;
        return this;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public BackpropagationTrainer setBatchSize(int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public BackpropagationTrainer setMomentum(float momentum) {
       this.momentum = momentum;
       return this;
    }

    public float getMomentum() {
        return momentum;
    }

    public float getLearningRate() {
        return learningRate;
    }
    
    public void stop() {
        stopTraining = true;
    }

    public float getTotalError() {
        return totalError;
    }

    public int getCurrentEpoch() {
        return epoch;
    }

    public OptimizerType getOptimizer() {
        return optimizer;
    }

    public BackpropagationTrainer setOptimizer(OptimizerType optimizer) {
        this.optimizer = optimizer;
        return this;
    }
    
}
