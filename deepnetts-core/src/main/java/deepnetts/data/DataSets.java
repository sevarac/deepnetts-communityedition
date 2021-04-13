/**
 *  DeepNetts is pure Java Deep Learning Library with support for Backpropagation
 *  based learning and image recognition.
 *
 *  Copyright (C) 2017  Zoran Sevarac <sevarac@gmail.com>
 *
 * This file is part of DeepNetts.
 *
 * DeepNetts is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <https://www.gnu.org/licenses/>.package
 * deepnetts.core;
 */

package deepnetts.data;

import deepnetts.data.preprocessing.scale.MaxScaler;
import deepnetts.data.preprocessing.scale.Standardizer;
import deepnetts.util.ColumnType;
import deepnetts.util.CsvFormat;
import java.io.File;
import java.io.IOException;
import deepnetts.util.DeepNettsException;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.regex.Pattern;
import javax.visrec.ml.data.DataSet;
import javax.visrec.ml.data.preprocessing.Scaler;


/**
 * Data set related utility methods.
 * 
 * @author zoran
 */
public class DataSets {

    public final static String DELIMITER_SPACE = " ";
    public final static String DELIMITER_COMMA = ",";
    public final static String DELIMITER_SEMICOLON = ";";
    public final static String DELIMITER_TAB = "\t";

   /**
     * Creates and returns data set from specified CSV file. Empty lines are
     * skipped
     *
     * @param csvFile CSV file
     * @param numInputs number of input values in a row
     * @param numOutputs number of output values in a row
     * @param hasColumnNames true if first row contains column names
     * @param delimiter delimiter used to separate values
     * @return instance of data set with values loaded from file
     *
     * @throws FileNotFoundException if file was not found
     * @throws IOException if there was an error reading file
     *
     * TODO: Detect if there are labels in the first line, if there are no
     * labels, set class1, class2, class3 in classifier evaluation! and detect
     * type of attributes Move this method to some factory class or something?
     * or as a default method in data set?
     *
     *  TODO: should I wrap IO with DeepNetts Exception?
     * Autodetetect delimiter; header and column type
     *
     */
    public static TabularDataSet readCsv(File csvFile, int numInputs, int numOutputs, boolean hasColumnNames, String delimiter) throws FileNotFoundException, IOException {
        TabularDataSet dataSet = new TabularDataSet(numInputs, numOutputs);
        BufferedReader br = new BufferedReader(new FileReader(csvFile));
        String line=null;
        // auto detect column names - ako sadrzi slova onda ima imena. Sta ako su atributi nominalni? U ovoj fazi se pretpostavlja d anisu...
        // i ako u redovima ispod takodje ima stringova u istoj koloni - detect header
        if (hasColumnNames) {    // get col names from the first line
            line = br.readLine().trim();
            String[] colNames = line.split(delimiter);
            // todo checsk number of col names
            dataSet.setColumnNames(colNames);
        } else {
            String[] colNames = new String[numInputs+numOutputs];
            for(int i=0; i<numInputs;i++)
                colNames[i] = "in"+(i+1);

            for(int j=0; j<numOutputs;j++)
                colNames[numInputs+j] = "out"+(j+1);

            dataSet.setColumnNames(colNames);
        }

        while ((line = br.readLine()) != null) {
           line = line.trim();
            if (line.isEmpty()) {
                continue; // skip empty lines
            }
            String[] values = line.split(delimiter);
            if (values.length != (numInputs + numOutputs)) {
                throw new DeepNettsException("Wrong number of values in the row " + (dataSet.size() + 1) + ": found " + values.length + " expected " + (numInputs + numOutputs));
            }
            float[] in = new float[numInputs];
            float[] out = new float[numOutputs];

            try {
                // these methods could be extracted into parse float vectors
                for (int i = 0; i < numInputs; i++) { //parse inputs
                    in[i] = Float.parseFloat(values[i]);
                }

                for (int j = 0; j < numOutputs; j++) { // parse outputs
                    out[j] = Float.parseFloat(values[numInputs + j]);
                }
            } catch (NumberFormatException nex) {
                throw new DeepNettsException("Error parsing csv, number expected line in " + (dataSet.size() + 1) + ": " + nex.getMessage(), nex);
            }

            dataSet.add(new TabularDataSet.Item(in, out));
        }

        return dataSet;
    }

    public static TabularDataSet readCsv(String fileName, int numInputs, int numOutputs, boolean hasColumnNames, String delimiter) throws IOException {
         return readCsv(new File(fileName), numInputs, numOutputs, hasColumnNames, delimiter);
    }

    public static TabularDataSet readCsv(String fileName, int numInputs, int numOutputs, boolean hasColumnNames) throws IOException {
        return readCsv(new File(fileName), numInputs, numOutputs, hasColumnNames, ",");
    }

    public static TabularDataSet readCsv(String fileName, int numInputs, int numOutputs, String delimiter) throws IOException {
        return readCsv(new File(fileName), numInputs, numOutputs, false, delimiter);
    }

    /**
     * Create data set from CSV file, using coma (,) as default delimiter and no
     * header (column names) in first row.
     *
     * @param fileName  Name of the CSV file
     * @param numInputs Number of input columns
     * @param numOutputs Number of output columns
     * @return
     * @throws IOException
     */
    public static TabularDataSet readCsv(String fileName, int numInputs, int numOutputs) throws IOException {
        return readCsv(new File(fileName), numInputs, numOutputs, false, ",");
    }

    // delimiter, hasHeader, column names and columnTypes
    public static CsvFormat detectCsvFormat(String fileName) throws FileNotFoundException, IOException {
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        String firstLine = br.readLine();

        // sta ako ima i navodnike ""

        // autodetect delimiter
        String delimiter = null;
        if (firstLine.contains(",")) delimiter = DELIMITER_COMMA;
        else if (firstLine.contains(";")) delimiter = DELIMITER_SEMICOLON;
        else if (firstLine.contains("\t")) delimiter = DELIMITER_TAB;
        else if (firstLine.contains(" ")) delimiter = DELIMITER_SPACE;  // da li je space delimiter za header?
        else throw new DeepNettsException("Unknown delimiter");

        boolean hasColumnNames = false;
        String[] columnNames = null;
        // da li prvi red sadrzi alfanumericka polja razdvojena delimiterima
        String[] firstLineFields = firstLine.split(delimiter);
        int colCount = firstLineFields.length;

        // mogu da budu i negativni brojevi!!!
        String intRegex = "^-?[0-9]+$"; //  "^-?(0|[1-9]\\d*)$"
        String decimalRegex = "^-?[0-9]+\\.[0-9]+$";    // "^\\d+(\\.\\d+)?$"
        String binaryRegex = "^[01]$";
        String numRegex = "^-?[0-9]+\\.?[0-9]+$";
        String alphaNumRegex = "^[a-zA-Z0-9_\\s\\-]+$"; // "^([a-zA-Z_0-9\\ \\-])+$"
        String alphaRegex = "^[a-zA-Z_\\s\\-]+$";

        boolean allNumeric = true;
        boolean allAlphaNum = true;

        // ako sadrzi sve samo numericke onda nema column names
        for(String field : firstLineFields) {
            boolean isNum = Pattern.matches(numRegex, field);
            boolean isAlphaNum = Pattern.matches(alphaNumRegex, field);
            allNumeric = allNumeric && isNum;
            allAlphaNum = allAlphaNum && isAlphaNum;
        }

        if (allNumeric) {
            hasColumnNames = false;
        } else if (allAlphaNum) { // most likely column names but might be also nominal
            columnNames = firstLineFields;
            hasColumnNames = true;
        } else { // mix of num and nominal most likely data row
            hasColumnNames = false;
        }

        // TODO: get next five rows and autodetect column types
        // int, dec, binary, string
        String[][] sampleRows=new String[5][colCount];
        for(int i=0; i<5; i++) {
            String line = br.readLine();
            String[] fields = line.split(delimiter);
            sampleRows[i] = fields; // todo trim all fields
        }

        // detect column types based on first 5 rows (or get random sample of 10 rows from first 100?)
        ColumnType colTypes[] = new ColumnType[colCount];
        for (int c=0; c<colCount; c++) {
            boolean allColsAlphaNum = true,
                    allColsBinary = true,
                    allColsDecimal = true,
                    allColsInt = true;

            for(int r=0; r<5; r++) {
                boolean isBinary = Pattern.matches(binaryRegex, sampleRows[r][c]);
                allColsBinary = allColsBinary && isBinary;

                boolean isInt = Pattern.matches(intRegex, sampleRows[r][c]);
                allColsInt = allColsInt && isInt;

                boolean isDecimal = Pattern.matches(decimalRegex, sampleRows[r][c]);
                allColsDecimal = allColsDecimal && ( isDecimal || isInt ); // moze da bud ekolona koje ima decimalne ali i int, ona se tretira kao decimalna

                boolean isAlphaNum = Pattern.matches(alphaNumRegex, sampleRows[r][c]);
                allColsAlphaNum = allColsAlphaNum && isAlphaNum;
            }

            if (allColsBinary) {
                colTypes[c] = ColumnType.BINARY;
            } else if (allColsInt) {
                colTypes[c] = ColumnType.INTEGER;
            } else if (allColsDecimal) {
                colTypes[c] = ColumnType.DECIMAL;
            } else {
                colTypes[c] = ColumnType.STRING;
            }

        }

        CsvFormat csvFormat = new CsvFormat();
        csvFormat.setDelimiter(delimiter);
        csvFormat.setColumnTypes(colTypes);
        csvFormat.setColumnNames(columnNames);
        csvFormat.setHasHeader(hasColumnNames);

        return csvFormat;

    }

    /**
     * Normalize specified data set and return used normalizer.
     * 
     * @param dataSet Data set to normalize
     * @return normalizer used to normalize data set
     */
    public static Scaler scaleMax(DataSet dataSet) {
        MaxScaler scaler = new MaxScaler(dataSet);
        scaler.apply(dataSet);
        
        return scaler;
    }
    
    public static Scaler standardize(DataSet dataSet) {
        Standardizer scaler = new Standardizer(dataSet);
        scaler.apply(dataSet);
        
        return scaler;
    }    

    // encode single row
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

    public static DataSet<?>[] trainTestSplit(DataSet<?> dataSet, double split) {
        dataSet.shuffle();
        return dataSet.split(split, 1-split);
    }

}
