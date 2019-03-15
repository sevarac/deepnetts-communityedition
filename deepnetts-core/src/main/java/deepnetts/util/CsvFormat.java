package deepnetts.util;

import java.util.Arrays;

public class CsvFormat {
    String delimiter;
    boolean hasHeader;
    String[] columnNames;
    ColumnType[] columnTypes;

    public String getDelimiter() {
        return delimiter;
    }

    public void setDelimiter(String delimiter) {
        this.delimiter = delimiter;
    }

    public boolean isHasHeader() {
        return hasHeader;
    }

    public void setHasHeader(boolean hasHeader) {
        this.hasHeader = hasHeader;
    }

    public String[] getColumnNames() {
        return columnNames;
    }

    public void setColumnNames(String[] columnNames) {
        this.columnNames = columnNames;
    }

    public ColumnType[] getColumnTypes() {
        return columnTypes;
    }

    public void setColumnTypes(ColumnType[] columnTypes) {
        this.columnTypes = columnTypes;
    }

    @Override
    public String toString() {
        return "CsvFormat{" + "delimiter=" + delimiter + ", hasHeader=" + hasHeader + ", columnNames=" + Arrays.toString(columnNames) + ", columnTypes=" + Arrays.toString(columnTypes) + '}';
    }




}
