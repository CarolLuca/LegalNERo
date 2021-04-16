package ro.racai.brat;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;

public class AnnotationWriter {
	private PrintWriter out;

	public AnnotationWriter(Path fpath) throws IOException {
		out=new PrintWriter(
				new BufferedWriter(
						new OutputStreamWriter(
								Files.newOutputStream(fpath, StandardOpenOption.WRITE, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
								,Charset.forName("UTF-8")
						)));		
	}
	
	public void close() throws IOException {
		out.close();
	}	
	
	public void write(Annotation ann) {
		out.println(ann.toString());
	}
	
	public void writeAll(List<Annotation> annList) {
		if(annList==null || annList.isEmpty())return ;
		
		for(Annotation ann:annList) {
			write(ann);
		}
	}
}
