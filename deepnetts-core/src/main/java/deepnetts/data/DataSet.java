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
