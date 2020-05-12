#include <iostream>

using namespace std;



class Node{
  public:
    int data;
    int min;
    Node*prev;
  public:
    Node(){
      prev = NULL;
    }
    Node(int val){
      data = val;
      prev = NULL;
    }
    Node(int val, int min){
      data = val;
      this->min = min;
    }
};

class StackWithMin{
  public:
    Node* top;
  public:
    void push(int val){
      Node* newNode = new Node(val);
      newNode->prev = top;
      if(top == NULL){
        newNode->min = val;
      }
      else{
        int min = (top->min < val)?top->min: val;
        newNode->min = min;
      }
      top = newNode;
    }
    Node pop(){
      Node temp = *top;
      delete top;
      top = temp.prev;
      return temp;
    }
    void min(){
      cout << top->min << endl;
    }
};

class SetOfStacks{
  public:
    StackWithMin stack[3];
    int numStacks = 0;
    int size = 0;
    int threshold = 2;
  public:
    void push(int val){
      if(size == threshold){
        ++numStacks;
        size = 0;
      }
      stack[numStacks].push(val);
      ++size;
    }
  void pop(){
    cout<<stack[numStacks].pop().data << endl;
    //cout << size << endl;
    size = size - 1;
    if (size<=0){
      --numStacks;
        size = threshold;
    }
  }
};
int main(){
  SetOfStacks ss;
  ss.push(1);
  ss.push(2);
  ss.push(3);
  ss.push(4);
  ss.pop();
  ss.pop();
  ss.pop();
  ss.pop();


}
