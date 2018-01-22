package deepnetts.examples;

import deepnetts.core.DeepNetts;
import deepnetts.data.ImageSet;
import deepnetts.eval.ConvolutionalClassifierEvaluator;
import deepnetts.net.ConvolutionalNetwork;
import deepnetts.net.layers.ActivationType;
import deepnetts.net.layers.OutputLayer;
import deepnetts.net.loss.LossType;
import deepnetts.net.train.BackpropagationTrainer;
import deepnetts.net.train.OptimizerType;
import deepnetts.util.DeepNettsException;
import deepnetts.util.FileIO;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

public class RunLegoPeople {
            
    int imageWidth = 96;
    int imageHeight = 96;

    String labelsFile = "/home/zoran/datasets/LegoPeopleNoviJecaPreprocessed/labels.txt";
    String trainingFile = "/home/zoran/datasets/LegoPeopleNoviJecaPreprocessed/train.txt";    
//    String labelsFile = "/home/zoran/datasets/legopeople2/labels.txt";
//    String trainingFile = "/home/zoran/datasets/legopeople2/train.txt";
   // String testFile = "/home/zoran/Desktop/LegoPeople/test.txt";         
    
    static final Logger LOG = Logger.getLogger(DeepNetts.class.getName());
    
    
    public void run() throws DeepNettsException, IOException {
     
        ImageSet imageSet = new ImageSet(imageWidth, imageHeight);
       
        LOG.info("Loading images...");
        
        imageSet.loadLabels(new File(labelsFile));
        imageSet.loadImages(new File(trainingFile), true);
        imageSet.invert();
        imageSet.zeroMean();
        imageSet.shuffle();
    
        
       ImageSet[] imageSets = imageSet.split(66, 34);
        
        LOG.info("Done loading images.");             
                
        // create convolutional neural network
        LOG.info("Creating neural network...");

        ConvolutionalNetwork legoPeopleNet = new ConvolutionalNetwork.Builder()
                                        .addInputLayer(imageWidth, imageHeight, 3) 
                                        .addConvolutionalLayer(5, 5, 1) 
                                        .addMaxPoolingLayer(2, 2, 2)                                  
                                        .addFullyConnectedLayer(30, ActivationType.TANH)  
                                        .addFullyConnectedLayer(10, ActivationType.TANH)  
                                        .addOutputLayer(1, ActivationType.SIGMOID)
                                        .withLossFunction(LossType.CROSS_ENTROPY)                
                                        .withRandomSeed(123)
                                        .build();        
              
        LOG.info("Done creating network.");       
        LOG.info("Training neural network..."); 
        
        legoPeopleNet.setOutputLabels(imageSet.getLabels());  
      //  List<ImageSet> subsets = imageSet.split(20, 80);
        
        // train convolutional network
        BackpropagationTrainer trainer = new BackpropagationTrainer(legoPeopleNet);
        trainer.setLearningRate(0.01f);
       // trainer.setMomentum(0.1f);
        trainer.setMaxError(0.06f);
        trainer.setOptimizer(OptimizerType.SGD);
        trainer.setBatchMode(true).setBatchSize(10);
        trainer.train(imageSets[0]);   
        
        LOG.info("Done training neural network."); 
          
        // save  network
        try {
            FileIO.writeToFile(legoPeopleNet, "legoPeople.net");
        } catch (IOException ex) {
            Logger.getLogger(RunLegoPeople.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        // deserialize and evaluate neural network
        ConvolutionalNetwork legoNet=null;
   //     try {
     //       legoNet = (ConvolutionalNetwork) FileIO.createFromFile("legoPeople.net");            
            ConvolutionalClassifierEvaluator recognitionTester = new ConvolutionalClassifierEvaluator();
            recognitionTester.evaluate(legoPeopleNet, imageSets[1]);     
            System.out.println(recognitionTester);                        
//        } catch (IOException | ClassNotFoundException ex) {
//            Logger.getLogger(RunLegoPeople.class.getName()).log(Level.SEVERE, null, ex);
//        }
                           
//        ImageRecognizer imageRecognizer = new DeepNettsImageRecognizer(legoPeopleNet);
//        List<RecognitionResult> results = imageRecognizer.recognize(new File("/home/zoran/datasets/LegoPeople/negative/bg1.jpg"));
//        System.out.println(results.toString());                       
    }
    
    
    
    
    public static void main(String[] args) {                                 
        try {
            (new RunLegoPeople()).run();
        } catch (DeepNettsException | IOException ex) {
            Logger.getLogger(RunLegoPeople.class.getName()).log(Level.SEVERE, null, ex);
        }
   
                
    }
}