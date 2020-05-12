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
    void pop(){
      cout << top->data << endl;
      top = top->prev;
    }
    void min(){
      cout << top->min << endl;
    }
};

int main(){
  StackWithMin stack;
  stack.push(1);
  stack.push(2);
  stack.push(-3);
  stack.pop();
  stack.min();

}
