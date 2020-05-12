#include <iostream>

using namespace std;

class Node{
public:
  int data;
  Node* leftChild;
  Node* rightChild;
   Node(int val){
    data = val;
    leftChild = NULL;
    rightChild = NULL;
  }
  
  void insertLeft(int val){
    Node* newNode = new Node(val);
    if(leftChild == NULL){
      leftChild = newNode;
    }
    else{
      newNode->leftChild = this->leftChild;
      this->leftChild = newNode;
    }
  }
  void insertRight(int val){
    Node* newNode = new Node(val);
    if(rightChild == NULL){
      rightChild = newNode;
    }
    else{
      newNode->rightChild = this->rightChild;
      this->rightChild = newNode;
    }
  }
};

int main(){
  Node test = Node(1);
  test.insertLeft(1);
  test.insertLeft(2);
  test.insertRight(3);
  test.insertRight(3);
  cout << test.data << test.leftChild->data << test.rightChild->data  << endl;
  return 1;
}
