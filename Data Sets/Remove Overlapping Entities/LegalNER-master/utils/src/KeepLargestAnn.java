import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

import ro.racai.brat.Annotation;
import ro.racai.brat.AnnotationReader;
import ro.racai.brat.AnnotationWriter;
import ro.racai.brat.MalformedAnnotationException;
import ro.racai.utils.Counter;

public class KeepLargestAnn {

	public static Counter cInitial=new Counter();
	public static Counter cRemaining=new Counter();
	
	public static void processFile(Path fpathIn) throws IOException, MalformedAnnotationException {
		AnnotationReader in = new AnnotationReader(fpathIn);
		List<Annotation> annotations=new ArrayList<>(100);
		for(Annotation ann=in.getNextAnnotation(); ann!=null; ann=in.getNextAnnotation()) {
			annotations.add(ann);
		}
		in.close();
		
		cInitial.inc(annotations.size());
		
		List<Annotation> newAnnotations=new ArrayList<>(annotations.size());
		for(Annotation ann:annotations) {
			boolean found=false;
			for(Annotation ann1:annotations) {
				if(ann==ann1)continue;
				
				//     XXXXXX
				// 1111111111111111
				if(ann1.getStart()<=ann.getStart() && ann.getEnd()<=ann1.getEnd()) {found=true; break;}
				
				//            XXXXXXXXXXX
				// 1111111111111111
				if(ann1.getStart()<=ann.getStart() && ann.getStart()<=ann1.getEnd() && ann.getSize()<ann1.getSize()) {
					found=true; break;
				}
				
                // XXXXXXXX
				//      1111111111111111
				if(ann1.getStart()<=ann.getEnd() && ann.getEnd()<=ann1.getEnd() && ann.getSize()<ann1.getSize()) {
					found=true; break;
				}
			}
			
			if(!found)newAnnotations.add(ann);
		}
		
		cRemaining.inc(newAnnotations.size());
		
		AnnotationWriter out=new AnnotationWriter(fpathIn);
		out.writeAll(newAnnotations);
		out.close();
	}
	
	public static void processFileNoException(Path fpathIn) {
		try {
			processFile(fpathIn);
		}catch(Exception ex) {
			System.out.println(String.format("[%s] %s", fpathIn.toString(),ex.getMessage()));
			ex.printStackTrace();
		}
	}
	
	
	public static void main(String[] args) throws Exception {
		if(args.length!=1) {
			System.out.println("KeepLargestAnn <path_to_annotated_files>");
			System.out.println("   files must have .ann extension");
			System.exit(-1);
		}
		
		System.out.println("Processing files in ["+args[0]+"]");
		
		Stream<Path> paths = Files.walk(Paths.get(args[0]),1);
		paths.parallel()
			.filter(Files::isRegularFile)
			.filter(x->(x.toString().endsWith(".ann")))
			.forEach(KeepLargestAnn::processFileNoException);
		paths.close();
		
		System.out.println("Initial annotations: "+cInitial.getC());
		System.out.println("Remaining annotations: "+cRemaining.getC());
	}	
	
}

