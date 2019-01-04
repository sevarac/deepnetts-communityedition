package deepnetts.data;

import static deepnetts.data.BasicDataSet.fromCSVFile;
import java.io.File;
import java.io.IOException;

/**
 *
 * @author zoran
 */
public class DataSets {
    public static DataSet readCsv(String fileName, int inputCount, int outputCount) throws IOException {
        return BasicDataSet.fromCSVFile(fileName, inputCount, outputCount);
    }
}
