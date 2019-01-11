package deepnetts.examples;

import java.io.FileNotFoundException;
import java.io.PrintWriter;

public class CsvFile {
    
    public static void write(double[][] data, String fileName) throws FileNotFoundException {
        PrintWriter pw = new PrintWriter(fileName);
        for(int i=0; i<data.length; i++) {
            pw.print(data[i][0]);
            pw.print(",");
            pw.println(data[i][1]);
        }
        pw.close();        
    }

        public static void write3D(double[][] data, String fileName) throws FileNotFoundException {
        PrintWriter pw = new PrintWriter(fileName);
        for(int i=0; i<data.length; i++) {
            pw.print(data[i][0]);
            pw.print(",");
            pw.print(data[i][1]);
            pw.print(",");
            pw.println(data[i][2]);            
        }
        pw.close();        
    }    
    
}
