package examplesbak;

import deepnetts.examples.util.CsvFile;
import deepnetts.examples.util.Plot;
import java.io.FileNotFoundException;

public class RandomLinearData21 {
    
    static final int X = 0;
    static final int Y = 1;
    static final int Z = 2;
    static final int NUM_POINTS = 30;
    static final int NOISE_FACTOR =15;  // smaller gives bigger noise
    
    // parameters of the linear function
    static double slope = 0.5;
    static double intercept = 0.2;
    
    public static void main(String[] args) throws FileNotFoundException {
                
        double[][] dataPoints = new double[NUM_POINTS][3];
        
        // generate random points with underlying linear trend as specified in method linear
        for(int i=0; i<dataPoints.length; i++) {
           dataPoints[i][X] =  0.5-Math.random();                     // X values are random numbers from [0, 1]
           dataPoints[i][Y] =  0.5-Math.random();                     // X values are random numbers from [0, 1]
           double noise = Math.random() / NOISE_FACTOR;          // generate random noise   
           dataPoints[i][Z] = linear(dataPoints[i][X], dataPoints[i][Y]) + noise; 
        }
                
        Plot.scatter(dataPoints);
        CsvFile.write3D(dataPoints, "linear_2_1.csv");
    }
        
    public static double linear(double x, double y) {
        double z = slope * x + slope * y + intercept;
        return z;
    }
    
}