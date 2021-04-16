#include <bits/stdc++.h>
using namespace std;
ifstream f("filename.txt");
map<string,int> file_name_index;
vector<string> rows[105];
struct cuvant
{
    int tip;
    int st;
    int dr;
}v[1005];
vector<cuvant> cuv[105][1005];
map<string,int> entity;
const int lim=1e5;
char ch[lim];
bool mycmp(cuvant x,cuvant y)
{
    return x.st<y.st;
}
bool usual(char ch)
{
    if(ch>='a' and ch<='z')
        return true;
    if(ch>='A' and ch<='Z')
        return true;
    if(ch>='0' and ch<='9') return true;
    if(ch=='\n') return true;
    if(ch=='.' or ch=='/' or ch==',' or ch==';'
    or ch==':' or ch=='-' or ch==' ' or ch=='$'
    or ch=='%' or ch=='^' or ch=='?' or ch=='!'
    or ch=='@' or ch=='*' or ch=='(' or ch==')'
    or ch=='[' or ch==']' or ch=='{' or ch=='}'
    or ch=='*' or ch=='#' or ch=='&' or ch=='>'
    or ch=='<' or ch=='\"' or ch=='\'' or ch=='|'
    or ch=='_' or ch=='+' or ch=='~' or ch=='`')
        return true;
    return false;
}
ofstream out("train_data_format.txt");
int main()
{
    entity["ORG"]=1;
    entity["LOC"]=2;
    entity["LAW"]=3;
    entity["PER"]=4;
    entity["TIME"]=5;
    string rev[7];
    rev[1]="ORG";
    rev[2]="LOC";
    rev[3]="LAW";
    rev[4]="PER";
    rev[5]="TIME";
    int cnt=0;
    string file_name;
    out<<"TRAIN_DATA=[";
    while(f>>file_name)
    {
        file_name_index[file_name]=++cnt;
        ifstream in(file_name);
        string gold_standoff=file_name;
        gold_standoff[0]='g';
        gold_standoff[1]='s';
        ifstream ing(gold_standoff);
        cout<<file_name<<'\n';
        string s;
        int nr=0;
        while(ing>>s)
        {
            ++nr;
            ing>>s;
            v[nr].tip=entity[s];
            ing>>v[nr].st>>v[nr].dr;
            //cout<<s<<' '<<v[nr].st<<' '<<v[nr].dr<<'\n';
            //v[nr].st--;
            //v[nr].dr-=2;
            ing.getline(ch,lim-2);
            /*string e="";
            int ind=0;
            while(ch[ind]==' ' or ch[ind]=='\t')
                ++ind;
            while(ch[ind]!='\n')
                e+=ch[ind],++ind;
            cout<<e.size()<<'\n';*/
            for(int i=1;i<strlen(ch)-1;++i)
            if(ch[i]=='"')
                ch[i]=' ';
            out<<"(\"";
            out<<(ch+1);
            out<<"\",{\"entities\":[";
            out<<"("<<0<<','<<strlen(ch)-1<<','<<"\""<<s<<"\")";
            out<<"]}),\n";
        }
        /*int unde=1;
        char ch;
        int cnte=0;
        string row="";
        int indice=-1,partial=0;
        sort(v+1,v+nr+1,mycmp);
        while(in.get(ch))
        {
            ++indice;
            if(ch!='\n' and !usual(ch))
            {
                string diac="";
                diac+=ch;
                while(in.get(ch) and !usual(ch))
                    diac+=ch;
                row+=diac;
                cout<<indice<<' '<<diac<<' '<<diac.size()<<'\n';
                indice+=diac.size();
            }
            cout<<indice<<' '<<ch<<'\n';
            if(ch!='\n' and usual(ch))
            {
                row+=ch;
                cnte=0;
            }
            else
            {
                //--indice;
                if(cnte>0 and row.size()>0)
                {
                    rows[cnt].push_back(row);
                    int elim=0,maxdr=0;
                    while(unde<=nr and v[unde].st<=indice)
                    {
                        ++elim;
                        v[unde].st-=partial;
                        v[unde].dr-=partial;
                        for(int j=v[unde].st;j<=v[unde].dr;++j)
                            cout<<row[j];
                        cout<<'\n';
                        if(elim>1 and unde>1 and v[unde].st<=maxdr)
                        {
                            maxdr=max(maxdr,v[unde].dr);
                            ++unde;
                            continue;
                        }
                        cuv[cnt][rows[cnt].size()].push_back(v[unde]);
                        maxdr=max(maxdr,v[unde].dr);
                        ++unde;
                    }
                    partial=indice+1;
                    row="";
                }
                cnte++;
            }
        }
        //return 0;
        */
    }
    out<<"]";
    /*out<<"TRAIN_DATA=[";
    for(int i=1;i<=100;++i)
    for(int j=0;j<rows[i].size();++j)
    {
        out<<"(\"";
        for(int k=0;k<rows[i][j].size();++k)
        {
            if(rows[i][j][k]!='\"')
                out<<rows[i][j][k];
            else out<<' ';
        }
        out<<"\",{\"entities\":[";
        bool ok=false;
        for(auto p:cuv[i][j+1])
        {
            if(ok)
                out<<',';
            out<<"("<<p.st<<','<<p.dr<<','<<"\""<<rev[p.tip]<<"\")";
            ok=true;
        }
        out<<"]}),\n";
    }
    out<<"]";*/
    return 0;
}
