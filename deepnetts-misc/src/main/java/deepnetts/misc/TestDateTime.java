package deepnetts.misc;

import static deepnetts.util.LicenseChecker.ISSUED_DATE;
import java.time.LocalDate;

public class TestDateTime {
    public static void main(String[] args) {
        String dateStr = "2020-03-10";
         LocalDate issueDate = LocalDate.parse(dateStr);
         System.out.println(issueDate);
    }
}
