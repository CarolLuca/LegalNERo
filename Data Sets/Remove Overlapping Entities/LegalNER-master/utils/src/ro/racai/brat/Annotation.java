package ro.racai.brat;

import java.io.Serializable;

public class Annotation implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1597378689881516510L;
	
	private String id;
	private String type;
	private int start;
	private int end;
	private String text;
	
	public Annotation(String id, String type, int start, int end, String text) {
		setId(id);
		setType(type);
		setStart(start);
		setEnd(end);
		setText(text);
	}
	
	public Annotation(String str) throws MalformedAnnotationException {
		String[] data=str.split("[\t]");
		if(data.length!=3)throw new MalformedAnnotationException("Invalid annotation ["+str+"]");
		
		setId(data[0]);
		setText(data[2]);
		
		String[] data1=data[1].split("[ ]");
		if(data1.length!=3)throw new MalformedAnnotationException("Invalid annotation ["+str+"]");
		
		setType(data1[0]);
		try {
			setStart(Integer.parseInt(data1[1]));
			setEnd(Integer.parseInt(data1[2]));
		}catch(Exception ex) {
			throw new MalformedAnnotationException("Invalid annotation ["+str+"]");
		}
	}
	
	public int getSize() {return getEnd()-getStart();}
	
	public String getId() {
		return id;
	}
	public void setId(String id) {
		this.id = id;
	}
	public String getType() {
		return type;
	}
	public void setType(String type) {
		this.type = type;
	}
	public int getStart() {
		return start;
	}
	public void setStart(int start) {
		this.start = start;
	}
	public int getEnd() {
		return end;
	}
	public void setEnd(int end) {
		this.end = end;
	}
	public String getText() {
		return text;
	}
	public void setText(String text) {
		this.text = text;
	}

	@Override
	public String toString() {
		return id + "\t" + type + " " + start + " " + end + "\t" + text;
	}
	
	
}
