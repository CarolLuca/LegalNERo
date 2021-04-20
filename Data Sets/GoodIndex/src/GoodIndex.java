import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class GoodIndex {
    public static void main(String args[]) throws FileNotFoundException {
        File titles=new File("src/filename.txt");
        Scanner read=new Scanner(titles);
        System.out.println("TRAIN_DATA=[");
        while(read.hasNextLine()){
            String origin_=read.nextLine();
            String listin_="listing_gs"+origin_.substring(2);
            File origin=new File("src/Original Texts/"+origin_);
            Scanner cin=new Scanner(origin);
            File listin=new File("src/Listings/"+listin_);
            Scanner lin=new Scanner(listin);
            String last=lin.nextLine();
            String etype=" ";
            while(cin.hasNextLine()){
                int cnt=0;
                String linie=cin.nextLine();
                if(linie.isEmpty())
                    continue;
                int n=linie.length();
                System.out.print("(\"");
                for(int i=0;i<n;i++){
                    if(linie.charAt(i)=='\"')
                        System.out.print(" ");
                    else System.out.print(linie.charAt(i));
                }
                System.out.println("\",{\"entities\":[");
                boolean modif=false;
                int last_ind=0;
                boolean stop=false;
                do {
                    // System.out.println(listin_);
                    modif=false;
                    int m=last.length(),ind=-1;
                    for(int i=0;i+m<=n;++i)
                        if(linie.substring(i,i+m).equals(last)==true){
                            if(lin.hasNextLine())
                                etype=lin.nextLine();
                            else stop=true;
                            if(lin.hasNextLine())
                                last=lin.nextLine();
                            else stop=true;
                            modif=true;
                            ind=i;
                            break;
                        }
                    if(ind>=last_ind && modif==true){
                    //if(modif==true){
                        last_ind=ind+m-1;
                        if(cnt>0)
                            System.out.print(",");
                        System.out.print("(");
                        System.out.print(ind);
                        System.out.print(",");
                        System.out.print(ind+m);
                        System.out.print(",");
                        System.out.print('\"'+etype+"\")");
                        ++cnt;
                    }
                }while(modif==true && stop==false);
                System.out.println("]}),");
            }
        }
        System.out.print("]");
    }
}
/*
TRAIN_DATA=[
("ORDIN nr. 1.615 din 19 noiembrie 2018",{"entities":[(0,37,"LAW")]}),
("19 noiembrie 2018",{"entities":[(0,17,"TIME")]}),
("Dolj",{"entities":[(0,4,"LOC")]}),
("MONITORUL OFICIAL",{"entities":[(0,17,"ORG")]})
]
 */
