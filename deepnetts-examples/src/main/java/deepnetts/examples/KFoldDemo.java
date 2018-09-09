package deepnetts.examples;

import deepnetts.data.DataSet;
import deepnetts.data.BasicDataSet;
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
        DataSet dataSet = BasicDataSet.fromCSVFile(new File("datasets/iris_data_normalised.txt"), 4, 3, ",");    
        //dataSet.setLabels(new String[] {"Setose", "Vrsicolor", "Virginica"});
        
        FeedForwardNetwork neuralNet = FeedForwardNetwork.builder()
                                            .addInputLayer(4)
                                            .addDenseLayer(20, ActivationType.TANH)
                                            .addOutputLayer(3, ActivationType.SOFTMAX)
                                            .withLossFunction(LossType.CROSS_ENTROPY)
                                            .withRandomSeed(123)
                                            .build();        
                
        BackpropagationTrainer trainer = new BackpropagationTrainer();
        trainer.setMaxError(0.01f);
        trainer.setLearningRate(0.5f);
        trainer.setMomentum(0.3f);
        trainer.setBatchMode(false);
        //trainer.setBatchSize(150);
        trainer.setMaxEpochs(10000);
        
        ClassifierEvaluator evaluator = new ClassifierEvaluator();
        
        KFoldCrossValidation kfcv = KFoldCrossValidation.builder()
                                        .withModel(neuralNet)
                                        .withDataSet(dataSet)
                                        .withTrainer(trainer)
                                        .withKFolds(5)
                                        .withEvaluator(evaluator)
                                        .build();
        
        PerformanceMeasure pm = kfcv.runCrossValidation();
        System.out.println(pm);
    }
    
}
