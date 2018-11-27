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
 * @author Zoran
 */
public interface DataSet <ITEM_TYPE extends DataSetItem> extends Iterable<ITEM_TYPE> {
    
    // TODO: remove idx, item, - as in List
    
    public void add(ITEM_TYPE item);
    
    public void addAll(DataSet<ITEM_TYPE> items);
    
    public ITEM_TYPE get(int index);
    
    public void clear();
    
    public boolean isEmpty();

    public int size(); 
    
    public DataSet[] split(int parts);
    
    public DataSet[] split(int ... parts); // float 0.65, 0.35
    
    public String[] getOutputLabels();
    
    public void setColumnNames(String[] labels);
    
    public void shuffle();
    
}
