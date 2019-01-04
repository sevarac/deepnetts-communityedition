package deepnetts.examples;

/**
 *
 * @author zoran
 */
public class GaussianScatterPlot {
    
    static final int X = 0;
    static final int Y = 1;    
    
    public static void main(String[] args) {
        double[][] data = new double[1000][2];
        
        data[0][X] = -4;
        data[0][Y] = gaussian(data[0][X], 0, 1);
        
//        for(int i=1; i<160; i++) {
//            data[i][X] = data[i-1][X] + 0.05;
//            data[i][Y] = gaussian(data[i][X], 0, 1)  ;
//        }
         
        for(int i=1; i<80; i++) {
            data[i][X] =  -4 + Math.random()*8;
            data[i][Y] = gaussian(data[i][X], 0, 1)  ;
        }
        
        Plot.scatter(data);         
    }
    
    public static double gaussian(double x, double mean, double std) {       
        double y = ( 1/Math.sqrt(2*Math.PI*std) ) * Math.exp(-((x-mean)*(x-mean))/ (2*std) );
        return y;
    }    
    
 
    
}
