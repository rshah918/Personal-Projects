#include <iostream>
#include <vector>
#include <unordered_set>

using namespace std;

class test{
  public:
    int data;
    test(){
      data = 0;
    }
    virtual int method1() = 0;//pure virtual
    virtual void method2(){
      cout << "parent" << endl;
    }
};

class child1: public test{
public:
  child1(){
  }

  int method1(){
    return 32;
  }
  void method3(){
    cout << "this is the child" << endl;
  }
};
int main(){
  child1 c1;
  c1.method2();
  return 1;
}
