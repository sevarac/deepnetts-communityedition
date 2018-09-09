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
package deepnetts.core;

import deepnetts.util.LicenseChecker;
import java.util.Properties;

/**
 * Global configuration and settings.
 *
 * @author Zoran Sevarac <zoran.sevarac@deepnetts.com>
 */
public final class DeepNetts {

    private static DeepNetts instance;
    
    // https://docs.oracle.com/javase/tutorial/essential/environment/sysprop.html
    private Properties prop;

    private DeepNetts() {
        // load properties from META-INF file that should be generated during the build
        prop = new Properties();
        prop.put("version", "1.0");
        prop.put("vendor", "Deep Netts LLC");
    }

    public static DeepNetts getInstance() {
        if (instance == null) {
            instance = new DeepNetts();
        }

        return instance;
    }

    public static void checkLicense() {
        LicenseChecker checker = new LicenseChecker();        
        try { 
            checker.checkLicense();       
        } catch(NullPointerException npe) {
            throw new LicenseChecker.LicenceException("License not found!");
        }
    }

    public String version() {
        return prop.getProperty("version");
    }
    
    public Properties getProperties() {
        return prop;
    }
    
}
