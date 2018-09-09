package deepnetts.examples;

import deepnetts.data.DataSet;
import deepnetts.data.BasicDataSet;
import deepnetts.data.BasicDataSetItem;
import deepnetts.data.DataSetItem;
import java.io.File;
import java.io.IOException;

/**
 * TODO: add breast cancer, mnist  and other UCI stuff
 * @author Zoran
 */
public class DataSets {

    public static DataSet iris() throws IOException {
       return BasicDataSet.fromCSVFile(new File("datasets/iris_data_normalised.txt"), 4, 3, ",");
    }

    public static DataSet xor() {
        DataSet dataSet = new BasicDataSet();

        DataSetItem item1 = new BasicDataSetItem(new float[] {0, 0}, new float[] {0});
        dataSet.add(item1);

        DataSetItem item2 = new BasicDataSetItem(new float[] {0, 1}, new float[] {1});
        dataSet.add(item2);

        DataSetItem item3 = new BasicDataSetItem(new float[] {1, 0}, new float[] {1});
        dataSet.add(item3);

        DataSetItem item4 = new BasicDataSetItem(new float[] {1, 1}, new float[] {0});
        dataSet.add(item4);

        return dataSet;
    }

    public static DataSet mnist() {
        return null;
    }
}
