package deepnetts.examples;

import deepnetts.data.DeepNettsBasicDataSet;

import deepnetts.data.DataSets;
import java.io.File;
import java.io.IOException;
import javax.visrec.ml.data.DataSet;
import deepnetts.data.DeepNettsDataSetItem;

/**
 * TODO: add breast cancer, mnist  and other UCI stuff
 * @author Zoran
 */
public class ExampleDataSets {

    public static DataSet iris() throws IOException {
       // TODO: apply some normalization here, as a param?
       return DataSets.readCsv("datasets/iris_data_normalised.txt", 4, 3);
    }

    public static DataSet xor() {
        DataSet dataSet = new DeepNettsBasicDataSet(2, 1);

        DeepNettsDataSetItem item1 = new DeepNettsBasicDataSet.Item(new float[] {0, 0}, new float[] {0});
        dataSet.add(item1);

        DeepNettsDataSetItem item2 = new DeepNettsBasicDataSet.Item(new float[] {0, 1}, new float[] {1});
        dataSet.add(item2);

        DeepNettsDataSetItem item3 = new DeepNettsBasicDataSet.Item(new float[] {1, 0}, new float[] {1});
        dataSet.add(item3);

        DeepNettsDataSetItem item4 = new DeepNettsBasicDataSet.Item(new float[] {1, 1}, new float[] {0});
        dataSet.add(item4);

        return dataSet;
    }


    public static DataSet mnist() {
        return null;
    }
}
