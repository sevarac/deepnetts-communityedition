package deepnetts.android;


import org.apache.logging.log4j.spi.LoggerContext;
import org.apache.logging.log4j.spi.LoggerContextFactory;

import java.net.URI;


/**
 * 
 * @author Krasimir Topchiyski
 * 
 * This factory can be used under Android in order to prevent the default Logging Factory to handle logging events.
 * The default factory uses Java JMX API which is not presented under Android and the code throws ClassNotFoundError.
 * You have to call LogManager.setFactory(new LoggerContextFactory_Dummy()); before any calls to Logger.getLogger(...).
 * The best place for LogManager.setFactory method call is during your android.app.Application instance onCreate method execution.
 */
public class LoggerContextFactory_Dummy implements LoggerContextFactory {
	

    /**
     * Creates a {@link LoggerContext}.
     *
     * @param fqcn            The fully qualified class name of the caller.
     * @param loader          The ClassLoader to use or null.
     * @param externalContext An external context (such as a ServletContext) to be associated with the LoggerContext.
     * @param currentContext  If true returns the current Context, if false returns the Context appropriate
     *                        for the caller if a more appropriate Context can be determined.
     * @return The LoggerContext.
     */
    @Override
    public LoggerContext getContext(String fqcn, ClassLoader loader, Object externalContext, boolean currentContext) {
        System.out.println("LoggerContextFactory_Dummy: getContext1");
        return new LoggerContext_Dummy();
    }

    /**
     * Creates a {@link LoggerContext}.
     *
     * @param fqcn            The fully qualified class name of the caller.
     * @param loader          The ClassLoader to use or null.
     * @param externalContext An external context (such as a ServletContext) to be associated with the LoggerContext.
     * @param currentContext  If true returns the current Context, if false returns the Context appropriate
     *                        for the caller if a more appropriate Context can be determined.
     * @param configLocation  The location of the configuration for the LoggerContext.
     * @param name            The name of the context or null.
     * @return The LoggerContext.
     */
    @Override
    public LoggerContext getContext(String fqcn, ClassLoader loader, Object externalContext, boolean currentContext, URI configLocation, String name) {
        System.out.println("LoggerContextFactory_Dummy: getContext2");
        return new LoggerContext_Dummy();
    }

    /**
     * Removes knowledge of a LoggerContext.
     *
     * @param context The context to remove.
     */
    @Override
    public void removeContext(LoggerContext context) {
        System.out.println("LoggerContextFactory_Dummy: removeContext");
    }
}
