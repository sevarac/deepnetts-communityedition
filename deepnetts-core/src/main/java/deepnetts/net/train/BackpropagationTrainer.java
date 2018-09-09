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
package deepnetts.net.train;

import deepnetts.core.DeepNetts;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.loss.LossFunction;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import org.apache.logging.log4j.LogManager;

/**
 * Backpropagation training algorithm for feed forward and convolutional neural networks.
 *
 * @see FeedForwardNetwork
 * @see ConvolutionalNetwork
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class BackpropagationTrainer {

    /**
     * Maximum training epochs. Training will stop when this number of epochs is
     * reached regardless the total network error.
     */
    private long maxEpochs = 100000L;

    /**
     * Maximum allowed error. Training will stop once total error has reached this value .
     */
    private float maxError = 0.01f;

    /**
     * Global learning rate
     */
    private float learningRate = 0.01f;

    /**
     * Optimizer type for all layers
     */
    private OptimizerType optimizer = OptimizerType.SGD;

    /**
     * Global momentum parameter
     */
    private float momentum = 0;

    /**
     * Set to true to use batch mode training
     */
    private boolean batchMode = false;
    
    /**
     * Size of mini batch. When full batch equals training set size
     */
    private int batchSize;
    
    private boolean stopTraining = false;

    /**
     * Current training epoch
     */
    private int epoch;
    
    
    private float totalError;
    
    /**
     * Shuffle training set before each epoch during training
     */
    private boolean shuffle = false; 

    
    private LossFunction lossFunction;

    private final List<TrainingListener> listeners = new ArrayList<>();

    public static final String PROP_MAX_ERROR       = "maxError";
    public static final String PROP_MAX_EPOCHS      = "maxEpochs";
    public static final String PROP_LEARNING_RATE   = "learningRate";
    public static final String PROP_MOMENTUM        = "momentum";
    public static final String PROP_BATCH_MODE      = "batchMode";
    public static final String PROP_BATCH_SIZE      = "batchSize";      // mini batch
    public static final String PROP_OPTIMIZER_TYPE  = "optimizerType";  // for mini batch

    private static final org.apache.logging.log4j.Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());

    public BackpropagationTrainer() {

    }

    public BackpropagationTrainer(Properties prop) {
        this.maxError = Float.parseFloat(prop.getProperty(PROP_MAX_ERROR));
        this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_EPOCHS));
        this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));
        this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
        this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));
        this.batchSize = Integer.parseInt(prop.getProperty(PROP_BATCH_SIZE));
        this.optimizer = OptimizerType.valueOf(prop.getProperty(PROP_OPTIMIZER_TYPE));
    }

    /**
     * This method does actual training procedure.
     *
     * Make this pure function so it can run in multithreaded - can train
     * several nn in parallel put network as param
     *
     * @param neuralNet
     * @param dataSet
     */
    public void train(NeuralNetwork neuralNet, DataSet<?> dataSet) {

        if (neuralNet == null) {
            throw new IllegalArgumentException("Parameter neuralNet cannot be null!");
        }
        if (dataSet == null) {
            throw new IllegalArgumentException("Argument dataSet cannot be null!");
        }
        if (dataSet.size() == 0) {
            throw new IllegalArgumentException("Data set is empty!");
        }

        neuralNet.setOutputLabels(dataSet.getOutputLabels());

        int trainingSamplesCount = dataSet.size();
        stopTraining = false;

        if (batchMode && (batchSize == 0)) {
            batchSize = trainingSamplesCount;
        }

        // set same lr to all layers!
        for (AbstractLayer layer : neuralNet.getLayers()) {
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
        float prevTotalError = 0, totalErrorChange;
        long startTraining, endTraining, trainingTime, startEpoch, endEpoch, epochTime;

        LOGGER.info("------------------------------------------------------------------------");
        LOGGER.info("TRAINING NEURAL NETWORK");
        LOGGER.info("------------------------------------------------------------------------");

        fireTrainingEvent(TrainingEvent.Type.STARTED);

        startTraining = System.currentTimeMillis();
        do {
            epoch++;
            lossFunction.reset();
            if (shuffle) {  // maybe remove this from gere, dont autoshuffle, for time series not needed
                dataSet.shuffle(); // dataset should be shuffled before each epoch http://ruder.io/optimizing-gradient-descent/index.html#adadelta
            }
            int sampleCounter = 0;

            startEpoch = System.currentTimeMillis();

            // maybe generate a sequence of random indexes instead of foreach, so i dontneed to shuffle in every epoch?
            for (DataSetItem dataSetItem : dataSet) { // for all items in dataset
                sampleCounter++;
                neuralNet.setInput(dataSetItem.getInput());   // set network input
                outputError = lossFunction.addPatternError(neuralNet.getOutput(), dataSetItem.getTargetOutput()); // get output error from loss function
                neuralNet.setOutputError(outputError); //mozda bi ovo moglao da bude uvek isti niz/reference pa ne mora da se seuje
                neuralNet.backward(); // do the backward propagation using current outputError - should I use outputError as a param here?

                // weight update for online mode after each training pattern
                if (!isBatchMode()) { // for online training update weight changes after each pass
                    neuralNet.applyWeightChanges();
                } else if (sampleCounter % batchSize == 0) { // mini batch
                    neuralNet.applyWeightChanges();
                    // do we need to reset lossFunction for mini batch?
                    float miniBatchError = lossFunction.getTotalValue();
                    LOGGER.info("Mini Batch:" + sampleCounter / batchSize + " Total Error:" + miniBatchError);
                    // da se ne ceka prvise dugo ako ima 60 000 slika nego da sve vreme prikazuje gresku
                }
                fireTrainingEvent(TrainingEvent.Type.ITERATION_FINISHED);
            }
            endEpoch = System.currentTimeMillis();

            //   batch weight update after entire data set - ako vrlicina dataseta nije deljiva sa batchSize - ostatak
            if (isBatchMode() && (trainingSamplesCount % batchSize != 0)) { // full batch. zarga ovaj gore ne pokriva?
                neuralNet.applyWeightChanges();
            }

            totalError = lossFunction.getTotalValue(); // - da li total error za ceo data set ili samo za mini  batch? lossFunction.getTotalError()

            totalErrorChange = totalError - prevTotalError; // todo: pamti istoriju ovoga i crtaj funkciju, to je brzina konvergencije na 10, 100, 1000 iteracija paterna - ovo treba meriti. Ovo moze i u loss funkciji
            prevTotalError = totalError;
            epochTime = endEpoch - startEpoch;

            // TODO; dont log only for epocsh but also for every batch
            LOGGER.info("Epoch:" + epoch + ", Time:" + epochTime + "ms, TotalError:" + totalError + ", ErrorChange:" + totalErrorChange); // EpochTime:"+epochTime + "ms,

            fireTrainingEvent(TrainingEvent.Type.EPOCH_FINISHED);

            stopTraining = ((epoch == maxEpochs) || (totalError <= maxError)) || stopTraining;

        } while (!stopTraining); // or learning slowed, or overfitting, ...

        endTraining = System.currentTimeMillis();
        trainingTime = endTraining - startTraining;

        LOGGER.info(System.lineSeparator() + "TRAINING COMPLETED");
        LOGGER.info("Total Training Time: " + trainingTime + "ms");
        LOGGER.info("------------------------------------------------------------------------");

        fireTrainingEvent(TrainingEvent.Type.STOPPED);
    }

    public long getMaxEpochs() {
        return maxEpochs;
    }

    public BackpropagationTrainer setMaxEpochs(long maxEpochs) {
        if (maxEpochs <= 0) throw new IllegalArgumentException("Max epochs should be greater then zero : " + maxEpochs);
        this.maxEpochs = maxEpochs;
        return this;
    }

    public float getMaxError() {
        return maxError;
    }

    
    public BackpropagationTrainer setMaxError(float maxError) {
        if (maxError < 0) throw new IllegalArgumentException("Max error cannot be negative : " + maxError);
        
        this.maxError = maxError;
        return this;
    }

    public BackpropagationTrainer setLearningRate(float learningRate) {
        if (learningRate < 0) throw new IllegalArgumentException("Learning rate cannot be negative : " + learningRate);
        if (learningRate > 1) throw new IllegalArgumentException("Learning rate cannot be greater then 1 : " + learningRate);
        
        this.learningRate = learningRate;
        
        return this;
    }

    public boolean getShuffle() {
        return shuffle;
    }

    public void setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
    }

    private void fireTrainingEvent(TrainingEvent.Type type) {
        for (TrainingListener l : listeners) {
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
