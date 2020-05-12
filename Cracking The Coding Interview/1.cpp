#include <iostream>
using namespace std;
bool isUniqueChars2(string str){
  bool char_set[256];
  for (int i = 0; i < str.length(); i++){
    int val = str[i];
    if(char_set[val]){
      return false;
    }
    char_set[val] = true;
  }
  return true;
}


class Node{
  public: Node * next;
  public: int data;

  public: Node(int val){
    this->data = val;
  }
};

int main(){

  bool out = isUniqueChars2("ABCC");
  cout << out << endl;
  return 1;
}
