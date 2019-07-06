package deepnetts.data;

/**
 * Generic interface for all data sets.
 * Data Set is an ordered collection of elements used to train a machine learning algorithm.
 *
 * TODO: implement DataSet from visrec-api
 * 
 * @author Zoran Sevarac
 * @param <ITEM_TYPE> type of elements in data set
 */
public interface DataSet <ITEM_TYPE extends DataSetItem> extends Iterable<ITEM_TYPE> {

    /**
     * Add data set item.
     * 
     * @param item
     */
    public void add(ITEM_TYPE item);

    public void addAll(DataSet<ITEM_TYPE> items);

    public ITEM_TYPE get(int idx);

    public void clear();

    public boolean isEmpty();

    public int size();

    public DataSet[] split(double ... parts);

    public DataSet[] split(int parts);

    /**
     * Randomly shuffle order of elements in dats set using global random generator/
     */
    public void shuffle();

    // these two methods below should be solved differently

    public String[] getOutputLabels();

    public void setColumnNames(String[] labels);

}
