#include <iostream>

using namespace std;
class Node{
  public:
    int value;
    Node * prev;
    Node(int val){
      this->value = val;
      prev = NULL;
    }
};

int fib(Node prev, int idx, int N){
  Node curr = Node(prev.value + prev.prev->value);
  curr.prev = &prev;
  idx++;
  if(idx == N){
    return curr.value;
  }
  else{
    return fib(curr, idx, N);
  }
}

int main(){
  Node n1 = Node(1);
  Node n2 = Node(1);
  n2.prev = &n1;
  cout<<fib(n2, 2, 28) << endl;
  return 1;
}
