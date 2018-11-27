/**  
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation 
 *  based learning and image recognition.
 * 
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 *  This file is part of DeepNetts.
 *
 *  DeepNetts is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.package deepnetts.core;
 */

package deepnetts.examples;

import deepnetts.data.DataSet;
import deepnetts.data.BasicDataSet;
import deepnetts.data.BasicDataSetItem;
import deepnetts.data.DataSetItem;
import java.io.File;
import java.io.IOException;

/**
 * TODO: add breast cancer, mnist  and other UCI stuff
 * @author Zoran Sevarac
 */
public class DataSets {

    public static DataSet iris() throws IOException {
       return BasicDataSet.fromCSVFile(new File("datasets/iris_data_normalised.txt"), 4, 3, ",");
    }
    
    // linear data set
    public static DataSet linear() throws IOException {
       return BasicDataSet.fromCSVFile(new File("datasets/iris_data_normalised.txt"), 4, 3, ",");
    }
    
    // plot?

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
