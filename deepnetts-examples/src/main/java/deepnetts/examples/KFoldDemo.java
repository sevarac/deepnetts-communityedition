package deepnetts.examples;

import deepnetts.data.DataSet;
import deepnetts.data.BasicDataSet;
import deepnetts.data.DataSets;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.PerformanceMeasure;
import deepnetts.net.FeedForwardNetwork;
import deepnetts.net.layers.activation.ActivationType;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.KFoldCrossValidation;
import java.io.File;
import java.io.IOException;

/**
 *
 * @author Zoran
 */
public class KFoldDemo {
    
    public static void main(String[] args) throws IOException {
        DataSet dataSet = DataSets.readCsv("datasets/iris_data_normalised.txt", 4, 3, true);
        //dataSet.setLabels(new String[] {"Setose", "Vrsicolor", "Virginica"});
        
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                                            .addInputLayer(4)
                                            .addDenseLayer(20, ActivationType.TANH)
                                            .addOutputLayer(3, ActivationType.SOFTMAX)
                                            .lossFunction(LossType.CROSS_ENTROPY)
                                            .randomSeed(123)
                                            .build();        
                
        BackpropagationTrainer trainer = new BackpropagationTrainer(neuralNet);
        trainer.setMaxError(0.01f);
        trainer.setLearningRate(0.5f);
        trainer.setMomentum(0.3f);
        trainer.setBatchMode(false);
        //trainer.setBatchSize(150);
        trainer.setMaxEpochs(10000);
        
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        
        // prilagodi ga https://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
        KFoldCrossValidation kfcv = KFoldCrossValidation.builder()
                                        .model(neuralNet)
                                        .dataSet(dataSet)
                                        .trainer(trainer)   // this one should be optional
                                        .splitsNum(5)
                                        .evaluator(evaluator)
                                        .build();
        
        // ispisi srednju vrednost i standardnu devijaciju po uzoru na jasona        
        PerformanceMeasure pm = kfcv.runCrossValidation();
        System.out.println(pm);
    }
    
}
