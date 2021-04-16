package ro.racai.utils;

public class Counter {

	private int c=0;

	public synchronized int getC() {
		return c;
	}

	public synchronized void setC(int c) {
		this.c = c;
	}
	
	public synchronized void inc() {
		this.c++;
	}
	
	public synchronized void inc(int n) {
		this.c+=n;
	}

}
