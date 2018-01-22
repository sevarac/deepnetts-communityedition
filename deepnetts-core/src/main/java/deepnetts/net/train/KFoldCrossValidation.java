package deepnetts.net.train;

import deepnetts.data.DataSet;
import deepnetts.eval.Evaluator;
import deepnetts.net.NeuralNetwork;

/**
 *
 * @author Zoran
 */
public class KFoldCrossValidation {
    
    // verovatno je najzgodnije stavit i builder za ovo posto ima vise od 3 stvari za konfiguraciju    
    
    int kFolds; //number of folds 5 or 10 most often used pg 184
    
    NeuralNetwork neuralNetwork; // arhitektura neuronske mreze
    BackpropagationTrainer trainer; // algoritam za trening sa svim svojim podesavanjima podesenim
    DataSet<?> dataSet; // data set koji se deli
    Evaluator evaluator; // mogao bi u logu da ispisuje rezultate evaluacije kao json
//    Evaluator<MeanSquaredError> evaluator; // returns PerformanceMeasure<MeanSquaredErrorLoss> ili da vracam kao properties/HashMap , da PerformanceMeasure ima konstante za razne vrste gresaka
    // PerformanceMeasure   da ima HashMap za vrednosti i konstante za kljuceve, tako da podrzava i custom performances measures. Standardne; MSE za regression, for classification accuracy, recall, precision, fscore
    
    //PerformanceMeasure<MeanSquaredErrorLoss> performanceMeasure; // MSE and avg Classification Perfromance
    
    // posto treba da radi multi threaded training, najbolje da ovo radi u istom thread-u, da ne bi rasipao threadove
    // najbolje da kreira trening i test set od foldova
        
    // podeli data set na k jednakih foldova
    // sa k-1 treniraj sa onim preostalim testiraj i izracunaj prosecne mere  performansi (MSE i klasifikacija)
    
    public void run() {
        
        final int foldSize = dataSet.size() / kFolds; // split data set into k equaly sized folds - i could also use split function here
        int itemIdx = 0;
                   
        DataSet[] folds = dataSet.split(60, 50);
        for (int testFold = 0; testFold < kFolds; testFold++) {
            for (int trainFold = 0; trainFold < kFolds; trainFold++) {
                DataSet trainingSet = new DataSet();
                DataSet testSet = new DataSet();

                if (trainFold != testFold) {
                    for (int i = 0; i < foldSize; i++) {
                        trainingSet.add(dataSet.get(itemIdx));
                        itemIdx++;
                    }
                } else {
                    for (int i = 0; i < foldSize; i++) {
                        testSet.add(dataSet.get(itemIdx));
                        itemIdx++;
                    }                    
                }
                
                trainer.train(trainingSet);
                evaluator.evaluate(neuralNetwork, testSet); // Peturn an instance of PerformanceMeaseure here
            }
        }        
        // get final evaluation results - avg performnce of all test sets
    }
    
    
}
