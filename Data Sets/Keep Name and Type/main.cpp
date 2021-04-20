#include <bits/stdc++.h>
using namespace std;
ifstream read("filename.txt");
const int lim=1e5+5;
char ch[lim];
int main()
{
    string name,type;
    while(read>>name)
    {
        name[0]='g';
        name[1]='s';
        ifstream in(name);
        ofstream out("listing_"+name);
        while(in>>name)
        {
            in>>type;
            in>>name;
            in>>name;
            in.getline(ch,lim);
            for(int i=1;i<strlen(ch);++i)
                out<<ch[i];
            out<<'\n';
            out<<type<<'\n';
        }
    }
    return 0;
}
