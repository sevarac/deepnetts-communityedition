package deepnetts.android;


import org.apache.logging.log4j.message.MessageFactory;
import org.apache.logging.log4j.spi.ExtendedLogger;
import org.apache.logging.log4j.spi.LoggerContext;

/**
 * 
 * @author Krasimir Topchiyski
 * 
 * @see deepnetts.android.LoggerContextFactory_Dummy comments in class level javadoc
 *
 */
class LoggerContext_Dummy implements LoggerContext {
    /**
     * An anchor for some other context, such as a ClassLoader or ServletContext.
     *
     * @return The external context.
     */
    @Override
    public Object getExternalContext() {
        return null;
    }

    /**
     * Retrieve an object by its name.
     *
     * @param key The object's key.
     * @since 2.13.0
     */
    @Override
    public Object getObject(String key) {
        return null;
    }

    /**
     * Store an object into the LoggerContext by name for later use.
     *
     * @param key   The object's key.
     * @param value The object.
     * @return The previous object or null.
     * @since 2.13.0
     */
    @Override
    public Object putObject(String key, Object value) {
        System.out.println("LoggerContext_Dummy: key=" + key + ", value=" + value);
        return null;
    }

    /**
     * Store an object into the LoggerContext by name for later use if an object is not already stored with that key.
     *
     * @param key   The object's key.
     * @param value The object.
     * @return The previous object or null.
     * @since 2.13.0
     */
    @Override
    public Object putObjectIfAbsent(String key, Object value) {
        return null;
    }

    /**
     * Remove an object if it is present.
     *
     * @param key The object's key.
     * @return The object if it was present, null if it was not.
     * @since 2.13.0
     */
    @Override
    public Object removeObject(String key) {
        return null;
    }

    /**
     * Remove an object if it is present and the provided object is stored.
     *
     * @param key   The object's key.
     * @param value The object.
     * @return The object if it was present, null if it was not.
     * @since 2.13.0
     */
    @Override
    public boolean removeObject(String key, Object value) {
        return false;
    }

    /**
     * Returns an ExtendedLogger.
     *
     * @param name The name of the Logger to return.
     * @return The logger with the specified name.
     */
    @Override
    public ExtendedLogger getLogger(String name) {
        return null;
    }

    /**
     * Returns an ExtendedLogger using the fully qualified name of the Class as the Logger name.
     *
     * @param cls The Class whose name should be used as the Logger name.
     * @return The logger.
     * @since 2.14.0
     */
    @Override
    public ExtendedLogger getLogger(Class<?> cls) {
        return null;
    }

    /**
     * Returns an ExtendedLogger.
     *
     * @param name           The name of the Logger to return.
     * @param messageFactory The message factory is used only when creating a logger, subsequent use does not change
     *                       the logger but will log a warning if mismatched.
     * @return The logger with the specified name.
     */
    @Override
    public ExtendedLogger getLogger(String name, MessageFactory messageFactory) {
        return null;
    }

    /**
     * Returns an ExtendedLogger using the fully qualified name of the Class as the Logger name.
     *
     * @param cls            The Class whose name should be used as the Logger name.
     * @param messageFactory The message factory is used only when creating a logger, subsequent use does not change the
     *                       logger but will log a warning if mismatched.
     * @return The logger.
     * @since 2.14.0
     */
    @Override
    public ExtendedLogger getLogger(Class<?> cls, MessageFactory messageFactory) {
        return null;
    }

    /**
     * Detects if a Logger with the specified name exists.
     *
     * @param name The Logger name to search for.
     * @return true if the Logger exists, false otherwise.
     */
    @Override
    public boolean hasLogger(String name) {
        return false;
    }

    /**
     * Detects if a Logger with the specified name and MessageFactory exists.
     *
     * @param name           The Logger name to search for.
     * @param messageFactory The message factory to search for.
     * @return true if the Logger exists, false otherwise.
     * @since 2.5
     */
    @Override
    public boolean hasLogger(String name, MessageFactory messageFactory) {
        return false;
    }

    /**
     * Detects if a Logger with the specified name and MessageFactory type exists.
     *
     * @param name                The Logger name to search for.
     * @param messageFactoryClass The message factory class to search for.
     * @return true if the Logger exists, false otherwise.
     * @since 2.5
     */
    @Override
    public boolean hasLogger(String name, Class<? extends MessageFactory> messageFactoryClass) {
        return false;
    }
}
