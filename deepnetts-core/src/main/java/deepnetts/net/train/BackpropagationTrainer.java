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

import deepnetts.net.train.opt.OptimizerType;
import deepnetts.core.DeepNetts;
import deepnetts.net.NeuralNetwork;
import deepnetts.net.layers.AbstractLayer;
import deepnetts.data.DataSet;
import deepnetts.data.DataSetItem;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.Evaluator;
import deepnetts.eval.PerformanceMeasure;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.loss.LossFunction;
import deepnetts.util.DeepNettsException;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Properties;
import org.apache.logging.log4j.LogManager;

/**
 * Backpropagation training algorithm for feed forward and convolutional neural networks.
 *
 * @see FeedForwardNetwork
 * @see ConvolutionalNetwork
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class BackpropagationTrainer implements Trainer, Serializable {

    /**
     * Maximum training epochs. Training will stop when this number of epochs is
     * reached regardless the total network error.
     * One epoch represents one pass of the entire training set.
     */
    private long maxEpochs = 100000L;

    /**
     * Maximum allowed error. Training will stop once total error has reached
     * this value .
     */
    private float maxError = 0.01f;

    /**
     * Global learning rate
     */
    private float learningRate = 0.01f;

    /**
     * Optimization algorithm type
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
     * Size of mini batch. When full batch is used, this equals training set size
     */
    private int batchSize;

    /**
     * Flag to stop training
     */
    private boolean stopTraining = false;

    /**
     * Current training epoch
     */
    private int epoch;
    
    /**
     * Value of loss function calculated on validation set
     */
    private float validationLoss=0, prevValidationLoss=0;
    
    private float accuracy=0;    

    private float totalTrainingLoss;

    /**
     * Shuffle training set before each epoch during training
     */
    private boolean shuffle = false;

    private NeuralNetwork<?> neuralNet;
    
    private transient DataSet<?> trainingSet;
    
    private transient DataSet<?> validationSet;
    
    private LossFunction lossFunction;
        
    /**
     * Use early stopping setting.
     */
    private boolean earlyStopping = false;

    /**
     * Checkpoint epochs for early stopping.
     */
    private int checkpointEpochs=1;
    
    private int prevCheckpointEpoch=0;
    
    /**
     * Min delta between checkpoints to continue training
     */
    private float checkpointMinDelta=0.000001f;
    
    private int checkpointNum = 2;  //  checkpoint limit
    private int checkpointCounter = 0; // checkpoint counter during training
    
    private float prevCheckpointTestLoss=100f;
        
    private String checkpointSavePath = "";    
    
    /**
     * Ovaj inicijalizovati ili na ovo ili na RMSE
     */
    private transient Evaluator<NeuralNetwork, DataSet<?>> eval = new ClassifierEvaluator();
    
    //regularization l1 or l2 add to loss 
    private float regL2, regL1;    
    // flag to save network weights during training
//    private boolean saveTrainingWeights = false;    
     

    
    
    private transient final List<TrainingListener> listeners = new ArrayList<>(); // TODO: add WeakReference for all listeners

    private static final org.apache.logging.log4j.Logger LOGGER = LogManager.getLogger(DeepNetts.class.getName());


    public BackpropagationTrainer(NeuralNetwork neuralNet) {
        this.neuralNet = neuralNet;
    }

    public BackpropagationTrainer(Properties prop) {
        // setProperties(prop); // all this should be done in setProperties
//        this.maxError = Float.parseFloat(prop.getProperty(PROP_MAX_ERROR));
//        this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_EPOCHS));
//        this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));
//        this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
//        this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));
//        this.batchSize = Integer.parseInt(prop.getProperty(PROP_BATCH_SIZE));
//        this.optimizer = OptimizerType.valueOf(prop.getProperty(PROP_OPTIMIZER_TYPE));
    }
    
    
    // da li da eksplicitno navedem validation set ili samo procenat trening koji se koristi za validaciju? odnos 2:1
    public void train(DataSet<?> trainingSet, DataSet<?> validationSet) {    
        this.validationSet = validationSet;
        train(trainingSet);
    }
     
    
    /**
     * This method does actual training procedure.
     *
     * Make this pure function so it can run in multithreaded - can train
     * several nn in parallel put network as param
     *
     * @param trainingSet training set data
     */
    public void train(DataSet<?> trainingSet) {

        if (trainingSet == null) {
            throw new IllegalArgumentException("Argument trainingSet cannot be null!");
        }
        if (trainingSet.size() == 0) {
            throw new IllegalArgumentException("Training set cannot be empty!");
        }

        this.trainingSet = trainingSet;
        neuralNet.setOutputLabels(trainingSet.getOutputLabels());

        int trainingSamplesCount = trainingSet.size();
        stopTraining = false;

        if (batchMode && (batchSize == 0)) {
            batchSize = trainingSamplesCount;
        }

        // set same lr to all layers!
        for (AbstractLayer layer : neuralNet.getLayers()) {
            layer.setLearningRate(learningRate);
            layer.setMomentum(momentum);
            layer.setRegularization(regL2);
            layer.setBatchMode(batchMode);
            layer.setBatchSize(batchSize);
            layer.setOptimizerType(optimizer);
        }

        lossFunction = neuralNet.getLossFunction();
        
        float[] outputError;
        epoch = 0;
        totalTrainingLoss = 0;
        float prevTotalLoss = 0, totalLossChange;
        long startTraining, endTraining, trainingTime, startEpoch, endEpoch, epochTime;

        LOGGER.info("------------------------------------------------------------------------");
        LOGGER.info("TRAINING NEURAL NETWORK");
        LOGGER.info("------------------------------------------------------------------------");

        fireTrainingEvent(TrainingEvent.Type.STARTED);

        startTraining = System.currentTimeMillis();
        do {
            epoch++;
            lossFunction.reset();
            validationLoss=0;
            accuracy=0;
            
            if (shuffle) {  // maybe remove this from gere, dont autoshuffle, for time series not needed - settings for Trainer
                trainingSet.shuffle(); // dataset should be shuffled before each epoch http://ruder.io/optimizing-gradient-descent/index.html#adadelta
            }
            int sampleCounter = 0;

            startEpoch = System.currentTimeMillis();

            // maybe generate a sequence of random indexes instead of foreach, so i dontneed to shuffle in every epoch?
            for (DataSetItem dataSetItem : trainingSet) { // for all items in trainng set
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
                    // do we need to reset lossFunction for mini batch - we probably should - saw it in tensorflow, ali ne ovde nego na pocetki batch-a
                    float miniBatchError = lossFunction.getTotal();
                    LOGGER.info("Mini Batch:" + sampleCounter / batchSize + " Batch Loss:" + miniBatchError);
                    // da se ne ceka prvise dugo ako ima 60 000 slika nego da sve vreme prikazuje gresku
                }
                fireTrainingEvent(TrainingEvent.Type.ITERATION_FINISHED); // BATCH_FINISHED?
                
                if (stopTraining) break; // if training was stoped externaly by calling stop() method
            }
            
           if (regL2!=0) lossFunction.addRegularizationSum(regL2 * neuralNet.getL2Reg()); // 0.00001f
            
            endEpoch = System.currentTimeMillis();

            //   batch weight update after entire data set - ako vrlicina dataseta nije deljiva sa batchSize - ostatak
            if (isBatchMode() && (trainingSamplesCount % batchSize != 0)) { // full batch. zarga ovaj gore ne pokriva?
                neuralNet.applyWeightChanges();
            }

            totalTrainingLoss = lossFunction.getTotal(); // - da li total error za ceo data set ili samo za mini  batch? lossFunction.getTotalError()

            totalLossChange = totalTrainingLoss - prevTotalLoss; // todo: pamti istoriju ovoga i crtaj funkciju, to je brzina konvergencije na 10, 100, 1000 iteracija paterna - ovo treba meriti. Ovo moze i u loss funkciji
            
            prevTotalLoss = totalTrainingLoss;
            
            // use validation set here instead of test set
            if (validationSet != null) {    // kako da znam da li je klasifikacija ili regresija? mozda da imam neki setting, flag?
                prevValidationLoss = validationLoss;   
                validationLoss = validationLoss(validationSet);   // pre je i test loss
                accuracy = testAccuracy(validationSet);// da li ovo da radim ovde ili na event. bolje ovde zbog sinhronizacije
            } else {
                accuracy = testAccuracy(this.trainingSet);  // ovo zameniti sa RMSE za regresiju i gore iznad // ako nije zadat valdiation set nemoj ni da pises?
            }
            
            epochTime = endEpoch - startEpoch;
            // todo: validation error change!!!
//            StringBuilder logMsgBuilder = new StringBuilder();
//            logMsgBuilder.append("Epoch:" + epoch);
//            logMsgBuilder.append(logMsgBuilder);
            
            if (validationSet != null)
                LOGGER.info( "Epoch:" + epoch + ", Time:" + epochTime + "ms, TrainError:" + totalTrainingLoss + ", TrainErrorChange:" + totalLossChange + ", ValidationError:" + validationLoss + ", Accuracy: "+accuracy);
            else
                LOGGER.info( "Epoch:" + epoch + ", Time:" + epochTime + "ms, TrainError:" + totalTrainingLoss + ", TrainErrorChange:" + totalLossChange + ", Accuracy: "+accuracy);
            
            
            if (Float.isNaN(totalTrainingLoss)) throw new DeepNettsException("NaN value during training!");
            
            fireTrainingEvent(TrainingEvent.Type.EPOCH_FINISHED);

            // EARLY STOPPING
            if (earlyStopping && (epoch > 0 && epoch % checkpointEpochs == 0)) {
                if (prevCheckpointTestLoss - validationLoss < checkpointMinDelta) {
                    if (checkpointCounter == checkpointNum) {
                        stop(); // stop if test change between two checkpoints is smaller than minDeltaLoss, (that is test loss is growing, stagnating or lowering too slow)
                    } else {
                        checkpointCounter++;    // count how many checkpoints have validation loss stagnated or grown?
                    }
                } else {
                    checkpointCounter = 0;  // reset counter for loss growth or stagnation
                }

                // save network at this checkpoint since loss if going down
                prevCheckpointTestLoss = validationLoss;
                prevCheckpointEpoch = epoch;
                try { // save to some tmp file only if test loss was smaller
                    FileIO.writeToFile(neuralNet, checkpointSavePath + File.separatorChar + "checkhpoint_NetworkTraining_epoch_" + epoch + ".dnet"); // TODO: use constant for extension
                } catch (IOException ex) {
                    LOGGER.catching(ex);
                }
            }
                                   
            stopTraining = stopTraining || ((epoch == maxEpochs) || (totalTrainingLoss <= maxError));

        } while (!stopTraining); // main training loop

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
        if (maxEpochs <= 0) {
            throw new IllegalArgumentException("Max epochs should be greater then zero : " + maxEpochs);
        }
        this.maxEpochs = maxEpochs;
        return this;
    }

    public float getMaxError() {
        return maxError;
    }

    public BackpropagationTrainer setMaxError(float maxError) {
        if (maxError < 0) {
            throw new IllegalArgumentException("Max error cannot be negative : " + maxError);
        }

        this.maxError = maxError;
        return this;
    }

    public BackpropagationTrainer setLearningRate(float learningRate) {
        if (learningRate < 0) {
            throw new IllegalArgumentException("Learning rate cannot be negative : " + learningRate);
        }
        if (learningRate > 1) {
            throw new IllegalArgumentException("Learning rate cannot be greater then 1 : " + learningRate);
        }

        this.learningRate = learningRate;

        return this;
    }
    
    public BackpropagationTrainer setL2Regularization(float regL2) {
        this.regL2 = regL2;
        return this;
    }    
    
    public BackpropagationTrainer setL1Regularization(float regL1) {
        this.regL1 = regL1;
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
            l.handleEvent(new TrainingEvent(this, type));
        }
    }

    public void addListener(TrainingListener listener) {
        Objects.requireNonNull(listener, "Training listener cannot be null!");
        
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

    public float getTrainingLoss() {
        return totalTrainingLoss;
    }

    public float getTestLoss() {
        return validationLoss;
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

    public DataSet<?> getTestSet() {
        return validationSet;
    }

    public void setTestSet(DataSet<?> testSet) {
        this.validationSet = testSet;
    }

    public boolean getEarlyStopping() {
        return earlyStopping;
    }

    public void setEarlyStopping(boolean earlyStopping) {
        this.earlyStopping = earlyStopping;
    }
    
    /**
     * Save weights during training on specified number of epochs
     * 
     * @param epochs 
     */
//    public void setSaveTrainingWeightsEpochs(int epochs) {
//        if (epochs == 0) {
//            this.saveTrainingWeights = false;
//        } else {
//            this.saveTrainingWeights = true;
//            this.saveTrainingWeightsEpochs = epochs;
//        }
//    }
    
    public BackpropagationTrainer setCheckpointSavePath(String path) {
        this.checkpointSavePath = path;
        return this;
    }

    public int getCheckpointEpochs() {
        return checkpointEpochs;
    }

    public BackpropagationTrainer setCheckpointEpochs(int checkpointEpochs) {
        this.checkpointEpochs = checkpointEpochs;
        return this;
    }

    public float getCheckpointMinDelta() {
        return checkpointMinDelta;
    }

    public BackpropagationTrainer setCheckpointMinDelta(float checkpointMinDelta) {
        this.checkpointMinDelta = checkpointMinDelta;
        return this;
    }

    public int getCheckpointNum() {
        return checkpointNum;
    }

    public BackpropagationTrainer setCheckpointNum(int checkpointNum) {
        this.checkpointNum = checkpointNum;
        return this;
    }

    
    
    /**
     * Sets properties from available keys in specified prop object.
     * 
     * @param prop 
     */    
    public void setProperties(Properties prop) {
        this.maxError = Float.parseFloat(prop.getProperty(PROP_MAX_ERROR));
        this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_EPOCHS));
        this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));
        this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
        this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));
        this.batchSize = Integer.parseInt(prop.getProperty(PROP_BATCH_SIZE));
        this.optimizer = OptimizerType.valueOf(prop.getProperty(PROP_OPTIMIZER_TYPE));        
        
        // iterate properties keys?use reflection to set them?
   //     this.maxLoss = Float.parseFloat(prop.getProperty(PROP_MAX_LOSS));
//        this.maxEpochs = Integer.parseInt(prop.getProperty(PROP_MAX_EPOCHS));

        if (prop.getProperty(PROP_LEARNING_RATE) != null)
            this.learningRate = Float.parseFloat(prop.getProperty(PROP_LEARNING_RATE));

//        this.momentum = Float.parseFloat(prop.getProperty(PROP_MOMENTUM));
//        this.batchMode = Boolean.parseBoolean(prop.getProperty(PROP_BATCH_MODE));
//        this.batchSize = Integer.parseInt(prop.getProperty(PROP_BATCH_SIZE));
//        this.optimizer = OptimizerType.valueOf(prop.getProperty(PROP_OPTIMIZER_TYPE));        
    }

    // property names
    public static final String PROP_MAX_ERROR = "maxError";
    public static final String PROP_MAX_EPOCHS = "maxEpochs";
    public static final String PROP_LEARNING_RATE = "learningRate";
    public static final String PROP_MOMENTUM = "momentum";
    public static final String PROP_BATCH_MODE = "batchMode";
    public static final String PROP_BATCH_SIZE = "batchSize";      // for mini batch
    public static final String PROP_OPTIMIZER_TYPE = "optimizer";  // for mini batch

    
    private float validationLoss(DataSet<? extends DataSetItem> validationSet) {
        lossFunction.reset();             
        float validationLoss =  lossFunction.valueFor(neuralNet, validationSet);
        return validationLoss;
    }
    
    // only for classification problems
    private float testAccuracy(DataSet<? extends DataSetItem> validationSet) {        
        PerformanceMeasure pm = eval.evaluatePerformance(neuralNet, validationSet);
        return pm.get(PerformanceMeasure.ACCURACY);
    }




}
