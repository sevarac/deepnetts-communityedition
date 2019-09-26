# deepnetts-communityedition
Deep Netts Engine Community Edition

Maven dependencies and Snapshot repository

    <dependencies>
        <dependency>
            <groupId>com.deepnetts</groupId>
            <artifactId>deepnetts-core-ce</artifactId>
            <version>1.1-SNAPSHOT</version>
            <type>jar</type>
        </dependency>
        <dependency>
            <groupId>javax.visrec</groupId>
            <artifactId>visrec-api</artifactId>
            <version>1.0-SNAPSHOT</version>
        </dependency>
        <dependency>
            <groupId>javax.visrec</groupId>
            <artifactId>visrec-ri</artifactId>
            <version>1.0-SNAPSHOT</version>
            <type>jar</type>
        </dependency>
    </dependencies>
    
    <repositories>
        <repository>
            <id>snapshots</id>
            <url>https://oss.sonatype.org/content/groups/public/</url>
            <snapshots>
                <enabled>true</enabled>
            </snapshots>
            <releases>
                <enabled>false</enabled>
            </releases>
        </repository>
    </repositories>  
