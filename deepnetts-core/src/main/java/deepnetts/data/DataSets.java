package deepnetts.data;

import java.io.File;
import java.io.IOException;
import static deepnetts.data.BasicDataSet.fromCsv;


/**
 *
 * @author zoran
 */
public class DataSets {
    public static DataSet readCsv(String fileName, int inputCount, int outputCount) throws IOException {
        return BasicDataSet.fromCsv(fileName, inputCount, outputCount);
    }
    
    public static DataSet readCsv(String fileName, int inputCount, int outputCount, boolean hasColumnNames) throws IOException {
        return BasicDataSet.fromCsv(fileName, inputCount, outputCount, hasColumnNames);
    }    
    
    public static DataSet readCsv(String fileName, int inputCount, int outputCount, String[] columnNames) throws IOException {
        BasicDataSet ds =  BasicDataSet.fromCsv(fileName, inputCount, outputCount);
        ds.setColumnNames(columnNames);
        return ds;
    }    
    
    public static DataSet normalizeMax(DataSet dataSet, boolean inplace) {
        // instantiate MaxNormalizer
        return null;
    }
    
    public static float[] oneHotEncode(final String label, final String[] labels) {   // different labels
        final float[] vect = new float[labels.length];
        // ako su brojeci i ako su stringovi, ako su sve nule, negative ...
        
        for(int i=0; i<labels.length; i++) {        
            if (labels[i].equals(label)) {
                vect[i] = 1;
            }   
        }
        // kako rsiti negative vektore?    
        return vect;
    }
    
//    public static float[] oneHotEncode(final int i, final int categories) {
//        
//    }    
    
//    public static DataSet random(int inputsNum, int outputsNum, inst size) {
//       
//        
//    }
    
    
}
