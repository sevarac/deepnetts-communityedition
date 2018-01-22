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
    
package deepnetts.eval;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.data.ExampleImage;
import deepnetts.data.ImageSet;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * TODO: put in visrec.ml
 * if class count == 2 use binary classifier else, its multi class classifier!
 * This class should build confusion matrix
 * 
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */                                         // Evaluator<Classifier>
public class ConvolutionalClassifierEvaluator implements Evaluator<ConvolutionalNetwork, ImageSet> { // use Classifier as a generic, wrap convolutional network with classifier
   
    /**
     * Class labels
     */
    private final List<String> classLabels = new ArrayList<>(); 
    
    // constants used as labels for  binary classification, maybe true/false?
    public final static String POSITIVE = "positive";
    public final static String NEGATIVE = "negative";
    
    /**
     * Classification stats for each class
     */
    HashMap<String, PerformanceMeasure> resultsByClass;
    // umesto ClassificationStats koristi PerformanceMeasure
    ConfusionMatrix confusionMatrix;
            
    /**
     * Total classification performance - should average existing
     */
  //  private ClassificationStats total;
    
    private float threshold = 0.5f; // this should go into classifier
    
              
    private void  init() {
        resultsByClass = new HashMap<>();
        
        if (classLabels.size() == 1) {
            confusionMatrix = new ConfusionMatrix(new String[] {POSITIVE, NEGATIVE} ); // labels for binary classification       
        } else {            
            classLabels.add(NEGATIVE);
            confusionMatrix = new ConfusionMatrix(classLabels.toArray(new String[classLabels.size()]));
            classLabels.forEach((label) -> {
                resultsByClass.put(label, new PerformanceMeasure());
            });                           
        }
    }
        
    @Override
    public Map<String, PerformanceMeasure>  evaluate(ConvolutionalNetwork convNet, ImageSet imageSet) {    
        classLabels.addAll(imageSet.getLabels());
        init();
                
        for(ExampleImage exampleImage : imageSet) {
            convNet.setInput(exampleImage.getInput());
            convNet.forward();
            float[] output = convNet.getOutput();
            processResult(output, exampleImage.getTargetOutput());                                 
        }        
        
       // calculatePercents();

        if (classLabels.size() == 1) {
            Map<String, PerformanceMeasure> perfMap = new HashMap();
            PerformanceMeasure pm = new PerformanceMeasure();

            int tp = confusionMatrix.getTruePositive();
            int tn = confusionMatrix.getTrueNegative();
            int fp = confusionMatrix.getFalsePositive();
            int fn = confusionMatrix.getFalseNegative();

            // ovo moze ovako za binarnu klasifikaciju, za multi class mora za svaku klasu posebno
            ClassificationMetrics cm = new ClassificationMetrics(tp, tn, fp, fn);

            pm.set("TotalClasses", classLabels.size());
            pm.set("TotalItems", cm.getTotal());

            pm.set("TruePositive", tp);
            pm.set("TrueNegative", tn);
            pm.set("FalsePositive", fp);
            pm.set("FalseNegative", fn);

            pm.set("TotalCorrect", tp + tn);
            pm.set("TotalIncorrect", fp + fn);

            pm.set(PerformanceMeasure.ACCURACY, cm.getAccuracy());
            pm.set(PerformanceMeasure.PRECISION, cm.getPrecision());
            pm.set(PerformanceMeasure.RECALL, cm.getRecall());
            pm.set(PerformanceMeasure.F1SCORE, cm.getF1Score());

             perfMap.put(classLabels.get(0), pm);
            return perfMap;
        } else {
           Map<String, PerformanceMeasure> perfMap = new HashMap();
           for(int clsIdx=0; clsIdx<classLabels.size(); clsIdx++) {
            PerformanceMeasure pm = new PerformanceMeasure();

            int tp = confusionMatrix.getTruePositive(clsIdx);
            int tn = confusionMatrix.getTrueNegative(clsIdx);
            int fp = confusionMatrix.getFalsePositive(clsIdx);
            int fn = confusionMatrix.getFalseNegative(clsIdx);

            // ovo moze ovako za binarnu klasifikaciju, za multi class mora za svaku klasu posebno
            ClassificationMetrics cm = new ClassificationMetrics(tp, tn, fp, fn);

            pm.set("TotalClasses", classLabels.size());
            pm.set("TotalItems", cm.getTotal());

            pm.set("TruePositive", tp);
            pm.set("TrueNegative", tn);
            pm.set("FalsePositive", fp);
            pm.set("FalseNegative", fn);

            pm.set("TotalCorrect", tp + tn);
            pm.set("TotalIncorrect", fp + fn);

            pm.set(PerformanceMeasure.ACCURACY, cm.getAccuracy());
            pm.set(PerformanceMeasure.PRECISION, cm.getPrecision());
            pm.set(PerformanceMeasure.RECALL, cm.getRecall());
            pm.set(PerformanceMeasure.F1SCORE, cm.getF1Score());
               
            perfMap.put(classLabels.get(clsIdx), pm);
           }
           return perfMap;
        }
    }

    // https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
    // http://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix
    private void processResult(float[] predictedOutput, float[] targetOutput) {
        
        if (classLabels.size() == 1) { // if its a binary classifier
            if ((predictedOutput[0] >= threshold) && (targetOutput[0] ==1)) {
                confusionMatrix.inc(0, 0); // tp is at [0, 0]
            } else if ((predictedOutput[0] < threshold) && (targetOutput[0] == 0)) {
                confusionMatrix.inc(1, 1); // tn is at [0, 0]
            } else if ((predictedOutput[0] >= threshold) && (targetOutput[0] ==0)) {
                confusionMatrix.inc(0, 1); // fp is at [0, 1]
            } else if ((predictedOutput[0] < threshold) && (targetOutput[0] == 1)) {
                confusionMatrix.inc(1, 0); // fn is at [1, 0]
            }            
        } else { // multi class classifier
            // nadji max iz predictedOutput i vidi da li je na istoj idx poziciji kao i 1 u targetOutput
            // da li dodati negative class u lavels/classes?
            int actualIdx = indexOfMax(targetOutput);
            String actualClass = null;
            
            if (!isNegativeTarget(targetOutput)) {
                actualClass = classLabels.get(actualIdx);
            } else {
                actualClass = NEGATIVE; // ako su svi nue, ond aje negativan primer
            }
            
            int predictedIdx = indexOfMax(predictedOutput); // ako su svi nule predictsIdx je od NEGATIVE
            String predictedClass = null;
            if (predictedOutput[predictedIdx] >= threshold) {
                predictedClass = classLabels.get(predictedIdx);
            } else {
                predictedClass = NEGATIVE;
            }
               
            confusionMatrix.inc(predictedIdx, actualIdx); 
            
            // todo sledece: add tp, fp, fn, tn here i to za svaku klasu posebno - vidi kao rade u pythonu to
            // mogu da imam jednu matricu n x n klasa
            // ilida za svaku klasu imam matricu 2x2 kao java ml
//            if (predictedIdx == actualIdx && predictedOutput[predictedIdx] > threshold) {
//                resultsByClass.get(actualClass).correct++; // todo sledece; ovde svaki treba da sadrzi confusion matrix ...!
//            } else if (predictedOutput[predictedIdx] > threshold && predictedIdx != actualIdx) {
//                resultsByClass.get(actualClass).incorrect++;  
//            } else if (actualClass.equals(NEGATIVE) && predictedClass.equals(NEGATIVE)) {
//                resultsByClass.get(actualClass).correct++;   
//            } else if (actualClass.equals(NEGATIVE) && !predictedClass.equals(NEGATIVE)) {
//                resultsByClass.get(actualClass).incorrect++;  
//            }
            
        }
    }
    
    
    /**
     * Returns index of max element in specified array
     * @param array
     * @return 
     */
    private int indexOfMax(float[] array) {
        int maxIdx = 0;
        for(int i=0; i<array.length; i++) {
            if (array[i] > array[maxIdx]) maxIdx = i;
        }
        return maxIdx;
    }
    
    private boolean isNegativeTarget(float[] array) {
        for(int i=0; i<array.length; i++)
            if (array[i]!=0) return false;
        
        return true;
    }

//    private void calculatePercents() {        
//        for(ClassificationStats stats : resultsByClass.values()) {
//            float totalForLabel = stats.correct + stats.incorrect;
//            stats.correctPercent = (stats.correct / totalForLabel)*100;
//            stats.incorrectPercent = (stats.incorrect / totalForLabel)*100;
//        }
//    }
    
    public static class ClassificationStats {
        String classLabel;
        int correct=0, incorrect=0;
        float correctPercent = 0, incorrectPercent = 0;
        
        int tp, tn, fp, fn;
        
        
        @Override
        public String toString() {
            return "correct = " + correct + " ("+correctPercent +"%), incorrect = " + incorrect+" ("+incorrectPercent+"%)";
        }                        
    }

    public float getThreshold() {
        return threshold;
    }

    public void setThreshold(float threshold) {
        this.threshold = threshold;
    }
        
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();

        sb.append(System.lineSeparator()).append("------------------------------------------------------------------------").append(System.lineSeparator()).
           append("CLASSIFIER EVALUATION RESULTS ").append(System.lineSeparator()).append("------------------------------------------------------------------------").append(System.lineSeparator());
        sb.append("Total classes: ").append(classLabels.size()).append(System.lineSeparator());
//        sb.append("Total correct: ").append(total.correct).append(System.lineSeparator());
//        sb.append("Total incorrect: ").append(total.incorrect).append(System.lineSeparator());        
        sb.append("Results by labels").append(System.lineSeparator());
        
        for(String label : resultsByClass.keySet()) {            
            PerformanceMeasure result = resultsByClass.get(label);
            if (result.get("TotalCorrect") == 0 && result.get("TotalIncorrect") == 0) continue; // if some of them is negative or nan dont show it
            sb.append(label).append(": ");   
            sb.append(result).append(System.lineSeparator());
        }
        
        return sb.toString();
    }
               
}