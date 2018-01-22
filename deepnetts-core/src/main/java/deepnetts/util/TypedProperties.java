package deepnetts.util;

import java.util.Properties;

/**
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public class TypedProperties extends Properties {
    
    public int getInt(String key) {
        return Integer.parseInt(getProperty(key));
    }
    
    public float getFloat(String key) {
        return Float.parseFloat(key);
    }
    
    public String getString(String key) {
        return getProperty(key).toString();
    }
    

}
