package deepnetts.net.train;

import deepnetts.data.BasicDataSet;
import deepnetts.data.DataSet;
import deepnetts.eval.ClassifierEvaluator;
import deepnetts.eval.Evaluator;
import deepnetts.eval.PerformanceMeasure;
import deepnetts.net.NeuralNetwork;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.lang3.SerializationUtils;

/**
 * Split data set into k parts of equal sizes (folds)
 * Train with k-1 folds, and test with 1 fold, repeat k timeas each with different test fld.
 *
 * https://svn.code.sf.net/p/java-ml/code/trunk/src/net/sf/javaml/classification/evaluation/CrossValidation.java
 * http://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold
 *
 * @author Zoran
 */
public class KFoldCrossValidation {

    private int splitsNum; //number of folds, typically  5 or 10 used, pg. 184    // numSplits
    private NeuralNetwork neuralNetwork; // arhitektura neuronske mreze
    private BackpropagationTrainer trainer; // algoritam za trening sa svim svojim podesavanjima podesenim
    private DataSet<?> dataSet; // data set koji se deli
    private Evaluator<NeuralNetwork, DataSet<?>> evaluator; // mogao bi u logu da ispisuje rezultate evaluacije kao json
    private final List<NeuralNetwork> trainedNetworks = new ArrayList<>();

    // posto treba da radi multi threaded training, najbolje da ovo radi u istom thread-u, da ne bi rasipao threadove
    // najbolje da kreira trening i test set od foldova

    // podeli data set na k jednakih foldova
    // sa k-1 treniraj sa onim preostalim testiraj i izracunaj prosecne mere  performansi (MSE i klasifikacija)

    public PerformanceMeasure runCrossValidation() {
        List<PerformanceMeasure> measures = new ArrayList<>();
        DataSet[] folds = dataSet.split(splitsNum);

        for (int testFoldIdx = 0; testFoldIdx < splitsNum; testFoldIdx++) {
            DataSet testSet = folds[testFoldIdx];
            DataSet trainingSet = new BasicDataSet(((BasicDataSet)dataSet).getInputsNum(), ((BasicDataSet)dataSet).getOutputsNum());
            trainingSet.setColumnNames( ((BasicDataSet)dataSet).getColumnNames());
            for (int trainFoldIdx = 0; trainFoldIdx < splitsNum; trainFoldIdx++) {
                if (trainFoldIdx == testFoldIdx) continue;
                trainingSet.addAll(folds[trainFoldIdx]);
            }

            // clone the original network each time before training - create a new instace that will be added to trainedNetworks
            NeuralNetwork neuralNet = SerializationUtils.clone(this.neuralNetwork);

            trainer.train(trainingSet); // napravi da trainer moze da sa istim parametrima pozove novu mrezu!!!!! ovo je problem, trainer zahteva novu instancu neuralNet ovde!!!
            PerformanceMeasure pe = evaluator.evaluatePerformance(neuralNet, testSet); // Peturn an instance of PerformanceMeaseure here
            measures.add(pe);
            trainedNetworks.add(neuralNet);
        }
        // get final evaluation results - avg performnce of all test sets - use some static method to get that
        return ClassifierEvaluator.averagePerformance(measures);
    }

    public List<NeuralNetwork> getTrainedNetworks() {
        return trainedNetworks;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {

        KFoldCrossValidation kFoldCV = new KFoldCrossValidation();

        public Builder splitsNum(int k) {
           kFoldCV.splitsNum = k;
           return this;
        }

        public Builder model(NeuralNetwork neuralNet) {
            kFoldCV.neuralNetwork = neuralNet;
            return this;
        }

        public Builder trainer(BackpropagationTrainer trainer) {
            kFoldCV.trainer = trainer;
            return this;
        }

        public Builder dataSet(DataSet dataSet) {
            kFoldCV.dataSet = dataSet;
            return this;
        }

        public Builder evaluator(Evaluator<NeuralNetwork, DataSet<?>> evaluator) {
            kFoldCV.evaluator = evaluator;
            return this;
        }

        public KFoldCrossValidation build() {
            return kFoldCV;
        }

    }
}
