#include <iostream>

using namespace std;

class Node{
  public:
    int value;
    Node * next;
    Node(int val){
      this->value = val;
      next = NULL;
    }
};
class LinkedList{
  Node * head;
  Node * tail;
  public:
    LinkedList(){
      head = NULL;
      tail = NULL;
    }
    void addNode(int val){
      Node * newNode = new Node(val);
    if(head == NULL){
      head = newNode;
      tail = newNode;
    }
    else{
      tail->next = newNode;
      tail = newNode;
    }
    Node * curr = head;
    while(curr != NULL){
      cout << curr->value << endl;
      curr = curr->next;
    }
    cout << "----------" << endl;
  }
  void insertNode(int val, int index){
    //index validation
    Node * curr = head;
    int count = 0;
    while(curr != NULL){
      if(count == index-1){
        break;
      }
      else{
        if(curr == tail){
          cout << "Invalid Index!" << endl;
          exit(1);
        }
        curr = curr->next;
        count++;
      }
    }
    //insert node after target
    Node * newNode = new Node(val);
    Node * next = curr->next;
    curr->next = newNode;
    newNode->next = next;

    //print LinkedList
    curr = head;
    while(curr != NULL){
      cout << curr->value << endl;
      curr = curr->next;
    }
    cout << "----------" << endl;
  }

};
int main(){

  LinkedList ll;
  ll.addNode(3);
  ll.addNode(5);
  ll.addNode(7);
  ll.addNode(9);
  ll.addNode(9);
  ll.insertNode(888,0);
  return 1;
}
