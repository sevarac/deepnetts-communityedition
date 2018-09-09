package deepnetts.util;

import com.javax0.license3j.licensor.HardwareBinder;
import com.javax0.license3j.licensor.License;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.time.LocalDate;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.bouncycastle.openpgp.PGPException;

/**
 * https://github.com/verhas/License3j/wiki/sample
 *
 * Public key must be in jar file, in META-INF, as specified by PUB_KEY
 * put license file in separate jar
 * how to specify digest? - hardcode from private key
 * @author Zoran Sevarac <zoran.sevaracc@deepnetts.com>
 */
public class LicenseChecker {

    private static final String LICENSE_FILE = "META-INF/deepnetts.lic";
    private static final String PUBLIC_KEY = "META-INF/DeepNettsPublicKey.gpg";
    
    // License properties
    public static final String REFERENCE_ID = "reference-id";
    public static final String LICENSE_TYPE = "license-type";
    public static final String ISSUED_DATE = "issued-date";
    public static final String VALID_UNTIL_DATE = "valid-until-date";
    public static final String DEVICE_ID = "device-id";
    public static final String COMPANY = "company";
    public static final String NAME = "name";
    public static final String EMAIL = "email";
    
    // this is probably signature only 64 bit? 8x8 bytes ...need stronger key/signature
    private final byte[] digest = new byte[]{
        (byte) 0xEF, (byte) 0x3E, (byte) 0x3D, (byte) 0xB1, (byte) 0x9B, (byte) 0x5E, (byte) 0x3F, (byte) 0xA3,
        (byte) 0x1F, (byte) 0x40, (byte) 0xB5, (byte) 0x2E, (byte) 0x8D, (byte) 0x6A, (byte) 0xBE, (byte) 0xAF,
        (byte) 0x11, (byte) 0x1E, (byte) 0x0F, (byte) 0xDA, (byte) 0x6C, (byte) 0xFD, (byte) 0x9C, (byte) 0x0A,
        (byte) 0x6B, (byte) 0x39, (byte) 0xCB, (byte) 0xBE, (byte) 0x47, (byte) 0x28, (byte) 0x60, (byte) 0x82,
        (byte) 0x7B, (byte) 0x87, (byte) 0xC9, (byte) 0x20, (byte) 0xC9, (byte) 0x39, (byte) 0x79, (byte) 0x3C,
        (byte) 0x83, (byte) 0xA3, (byte) 0x1A, (byte) 0x89, (byte) 0x9B, (byte) 0x31, (byte) 0xFB, (byte) 0x42,
        (byte) 0x4E, (byte) 0x22, (byte) 0xB2, (byte) 0xF9, (byte) 0x9A, (byte) 0x62, (byte) 0x5B, (byte) 0xFB,
        (byte) 0x3B, (byte) 0x34, (byte) 0x03, (byte) 0x73, (byte) 0xE2, (byte) 0xF6, (byte) 0xE7, (byte) 0x7E
    };
    
    private License license;
    

    public boolean checkLicense() {       
        try {
            license = new License();
            license.loadKeyRingFromResource(PUBLIC_KEY, digest); // ne nalazi javni kljuc
            license.setLicenseEncoded(this.getClass().getClassLoader().getResourceAsStream(LICENSE_FILE), "utf-8");

            checkDate();

            // check hardware biniding
            final HardwareBinder hb = new HardwareBinder();
            if (!license.getFeature(DEVICE_ID).isEmpty() && !license.getFeature(DEVICE_ID).equals(hb.getMachineIdString())) {
                throw new LicenceException("License not licensed to this device");
            }

            return true;

        } catch (IOException ex) {
            Logger.getLogger(LicenseChecker.class.getName()).log(Level.SEVERE, null, ex);
            return false;
        } catch (PGPException ex) {
            Logger.getLogger(LicenseChecker.class.getName()).log(Level.SEVERE, null, ex);
            return false;
        }
    }

    protected void checkDate() {
        LocalDate issueDate = LocalDate.parse(license.getFeature(ISSUED_DATE));
        LocalDate today = LocalDate.now();
        if (today.isBefore(issueDate)) {
            throw new LicenceException("Issue date is too late, probably tampered system time");
        }

        LocalDate validUntilDate = LocalDate.parse(license.getFeature(VALID_UNTIL_DATE));
        if (today.isAfter(validUntilDate)) {
            throw new LicenceException("License expired.");
        }
    }
    
    public Properties getLicenseProperties() {
        Properties prop = new Properties();
        
        prop.put(REFERENCE_ID, license.getFeature(REFERENCE_ID));
        prop.put(LICENSE_TYPE, license.getFeature(LICENSE_TYPE));
        prop.put(ISSUED_DATE, license.getFeature(ISSUED_DATE));
        prop.put(VALID_UNTIL_DATE, license.getFeature(VALID_UNTIL_DATE));
        prop.put(COMPANY, license.getFeature(COMPANY));
        prop.put(NAME, license.getFeature(NAME));
        prop.put(EMAIL, license.getFeature(EMAIL));
        prop.put(DEVICE_ID, license.getFeature(DEVICE_ID));
        
        return prop;
    }
    
    public void printLicense() {
        System.out.println(REFERENCE_ID + " : " + license.getFeature(REFERENCE_ID));
        System.out.println(LICENSE_TYPE + " : " + license.getFeature(LICENSE_TYPE));        
        System.out.println(ISSUED_DATE + " : " + license.getFeature(ISSUED_DATE));
        System.out.println(VALID_UNTIL_DATE + " : " + license.getFeature(VALID_UNTIL_DATE));
        System.out.println(COMPANY + " : " + license.getFeature(COMPANY));
        System.out.println(NAME + " : " + license.getFeature(NAME));     
        System.out.println(EMAIL + " : " + license.getFeature(EMAIL));    
        System.out.println(DEVICE_ID + " : " + license.getFeature(DEVICE_ID));             
//        System.out.println("Signature: ");             
//        System.out.println(digest);             
    }

    // u klasu DeepNetts ubaciti staticku metodu DeepNetts.checkLicense() koja instancira license checkera  -- koja se poziva u konstruktorima
    // bilje da licenca bude van jar-a da ne moram da za svakog klijenta buildujem poseban jar
    // ali sama licenca moze dabude jar fajl? deepNettsLicense.jar ??? to bi bilo odlicno takoje stavim na class path
    // u paket deepnetts.license u root?
    // Device id: 856017b3-c715-3383-91fc-2bd7e1541f2d    
    public static void main(String[] args) throws UnsupportedEncodingException, SocketException, UnknownHostException {
        LicenseChecker checker = new LicenseChecker();

        try {
            checker.checkLicense();
            System.out.println("License is valid.");
        } catch(LicenceException lex) {
            System.out.println("License is not valid.");
        }
        
        checker.printLicense();        

    }

    public static class LicenceException extends RuntimeException {

        public LicenceException(String message) {
            super(message);
        }

    }

}
