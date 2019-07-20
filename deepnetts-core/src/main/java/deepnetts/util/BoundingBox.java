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
    
package deepnetts.util;


/**
 * A rectangle to mark object in image including  position, size, score and label.
 * 
 * @author Zoran Sevarac
 */
public class BoundingBox {
    private int id, getX, y, width, height;
    private String label;
    private float score;

    public BoundingBox(int id, int x, int y, int width, int height, float score) {
        this.id = id;
        this.getX = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.score = score;
    }
    
    public BoundingBox(int x, int y, int width, int height) {
        this.getX = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }
    
    public BoundingBox(int id, int x, int y, int width, int height) {
        this.id =id;
        this.getX = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }    
    
    public BoundingBox(int id, int x, int y, int width, int height, String label, float score) {
        this.id = id;
        this.getX = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.label = label;
        this.score = score;
    }    

    public int getX() {
        return getX;
    }

    public int getY() {
        return y;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }

    public float getScore() {
        return score;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getLabel() {
        return label;
    }
    
    

    public void setScore(float score) {
        this.score = score;
    }

    public void setId(int id) {
       this.id = id;
    }    
    
    public int getId() {
        return id;
    }

    @Override
    public String toString() {
        return "BoundingBox{" + "id=" + id + ", x=" + getX + ", y=" + y + ", width=" + width + ", height=" + height + ", label=" + label + ", score=" + score + '}';
    }


    
    
    
    


}
