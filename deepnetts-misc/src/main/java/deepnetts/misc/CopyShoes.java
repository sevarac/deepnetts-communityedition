package deepnetts.misc;

import deepnetts.util.DeepNettsException;
import deepnetts.util.ImageSetUtils;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;

public class CopyShoes {

    public static void main(String[] args) throws IOException {
        String srcPath = "D:\\datasets\\ut-zap50k-images-square";   // Boots, Sandals, Shoes, Slippers

        ImageSetUtils.createImageIndex("D:\\datasets\\ut-zap50k-images-square");
        
       // Morao bih ovo rekurzivno da iteriram dublje
       List<String> allFiles = new ArrayList();
        createFileIndex(allFiles, srcPath);
        System.out.println(allFiles);
        
        //
        
        // izlistaj sve fajlove
        // onda ih kopiraj u 4 osnovna direktorijuma        
                
    }
    
    public static void createFileIndex(List<String> files, String path) throws IOException { // provide a path to train or text dir
                String destPath = "D:\\datasets\\ut-zap50k-images-square-mixed";
        File rootDir = new File(path);
        if (!rootDir.isDirectory()) {
            throw new DeepNettsException("Specified path must be a directory: " + path);
        }

        String[] fileList = rootDir.list();
        for (String filename : fileList) {
            String fullPath = path + "\\" + filename;
            File f = new File(fullPath);
            if (!f.isDirectory()) {
                files.add(fullPath);
                
                if (fullPath.contains("Sandals")) Files.copy(Paths.get(fullPath), Paths.get(destPath +"\\Sandals\\"+filename), StandardCopyOption.REPLACE_EXISTING);
                    else if (fullPath.contains("Shoes")) Files.copy(Paths.get(fullPath), Paths.get(destPath +"\\Shoes\\"+filename), StandardCopyOption.REPLACE_EXISTING);
                    else if (fullPath.contains("Boots")) Files.copy(Paths.get(fullPath), Paths.get(destPath +"\\Boots\\"+filename), StandardCopyOption.REPLACE_EXISTING);
                    else if (fullPath.contains("Slippers")) Files.copy(Paths.get(fullPath), Paths.get(destPath +"\\Slippers\\"+filename), StandardCopyOption.REPLACE_EXISTING);                
                
            } else {
                createFileIndex(files, fullPath);
            }
            

        }
    }
    
}
