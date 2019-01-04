package deepnetts.examples;

import java.io.FileNotFoundException;

public class RandomLinearData {
    
    static final int X = 0;
    static final int Y = 1;
    static final int NUM_POINTS = 20;
    static final int NOISE_FACTOR =5;  // smaller gives bigger noise
    
    // parameters of the linear function
    static double slope = 0.5;
    static double intercept = 0.2;
    
    public static void main(String[] args) throws FileNotFoundException {
                
        double[][] dataPoints = new double[NUM_POINTS][2];
        
        // generate random points with underlying linear trend as specified in method linear
        for(int i=0; i<dataPoints.length; i++) {
           dataPoints[i][X] =  Math.random();                     // X values are random numbers from [0, 1]
           double noise = Math.random() / NOISE_FACTOR;          // generate random noise   
           dataPoints[i][Y] = (linear(dataPoints[i][X]) + noise)/2; // add noise to underlying linear function
        }
                
        Plot.scatter(dataPoints);    
        CsvFile.write(dataPoints, "linear.csv");          
    }
        
    public static double linear(double x) {
        double y = slope * x + intercept;
        return y;
    }
    
}