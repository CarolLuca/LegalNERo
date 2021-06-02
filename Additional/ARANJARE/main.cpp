#include <bits/stdc++.h>
using namespace std;
ifstream in("date.in");
ofstream out("date.out");
const int lim=1e6+4;
char ch[lim];
int main()
{
    int linie=0;
    for(;in.getline(ch,lim-3);)
    {
        ++linie;
        if(strlen(ch)!=0)
        {
            for(int i=0;i<strlen(ch);++i)
            if(ch[i]!='\n' and ch[i]!='\r')
                out<<ch[i];
            in.getline(ch,lim-3);
            for(int i=0;i<strlen(ch);++i)
            if(ch[i]!='\n' and ch[i]!='\r')
                out<<ch[i];
            if(linie>1000)
                out<<'\n',
                linie=0;
        }
    }
    return 0;
}
