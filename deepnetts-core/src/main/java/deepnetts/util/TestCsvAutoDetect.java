package deepnetts.util;

import deepnetts.data.DataSets;
import java.io.IOException;

public class TestCsvAutoDetect {
    public static void main(String[] args) throws IOException {
        CsvFormat format = DataSets.detectCsvFormat("iris.data-normalized.txt");
        System.out.println(format);
    }
}
