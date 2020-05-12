#include <iostream>

using namespace std;

class Node{
  public:
    int data;
    Node * next;
  public:
    Node(int val){
      data = val;
      next = NULL;
  }
};
class Stack{
  Node * top;
  public:
    Stack(){
      top = NULL;
    }
    void push(int val){
      if(top == NULL){
        Node* newNode = new Node(val);
        top = newNode;
      }
      else{
        Node* newNode = new Node(val);
        newNode->next = top;
        top = newNode;
      }
    }
    void show(){
      Node curr = *top;
      while(curr.next != NULL){
        cout << curr.data << endl;
        curr = *curr.next;
      }
    }
    int pop(){
      Node* newNode = top;
      top = top->next;
      return newNode->data;
    }


};


int main(){
  Stack teststack = Stack();
  teststack.push(1);
  teststack.push(2);
  teststack.push(3);
  teststack.push(4);
  //teststack.show();
  cout << teststack.pop() << endl;
  cout << teststack.pop() << endl;
  cout << teststack.pop() << endl;
  cout << teststack.pop() << endl;

}
