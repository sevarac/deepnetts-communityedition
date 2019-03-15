package deepnetts.util;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RegexTestRun {
    public static void main(String[] args) {
        String intRegex = "^-?[0-9]+$";
        String decRegex = "^-?[0-9]+\\.[0-9]+$";
        String binaryRegEx = "^[01]$";
        String numRegex = "^-?[0-9]+\\.?[0-9]+$";
        String alphaNumRegex = "^[a-zA-Z0-9_]+$";
        String alphaRegex = "^[a-zA-Z_]+$";

        // kako i negativni?

        String str="10.5";

        boolean b = Pattern.matches(decRegex, str);

        System.out.println( b);

    }
}
