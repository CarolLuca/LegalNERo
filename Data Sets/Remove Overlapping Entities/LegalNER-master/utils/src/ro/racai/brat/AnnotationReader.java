package ro.racai.brat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public class AnnotationReader {
	private BufferedReader in;

	public AnnotationReader(Path fpath) throws IOException {
		in=new BufferedReader(new InputStreamReader(Files.newInputStream(fpath, StandardOpenOption.READ),Charset.forName("UTF-8")));
	}
	
	public void close() throws IOException {
		if(in!=null) {
			in.close();
			in=null;
		}
	}
	
	public Annotation getNextAnnotation() throws IOException, MalformedAnnotationException {
		if(in==null)return null;
		
		for(String line=in.readLine();;line=in.readLine()) {
			if(line==null) {
				close();
				return null;
			}
			
			if(line.isEmpty() || line.startsWith("#"))continue;
			
			return new Annotation(line);
		}
		
		
	}

}
