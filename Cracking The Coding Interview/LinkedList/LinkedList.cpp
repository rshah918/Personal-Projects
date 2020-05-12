#include <iostream>

using namespace std;


//Node class
class LinkedList{
public:
  struct Node{
    public:
      int data;
      Node* next;
      Node(){
        next = NULL;
      }
  };
Node *head;
Node *tail;

LinkedList(){
  this->head = NULL;
  this->tail = NULL;
}

  //add node
  void addNode(int data){
    //If LL is empty, make head and tail point to the node
    Node *newNode;
    newNode = new Node;
    newNode->data = data;
    if (head == NULL and tail == NULL){

      head = newNode;
      tail = newNode;
    }
    //This function only adds nodes at the end of the linked list, not at a particular location
    //Thus, set the tails "next" variable == to the node you want to add
      //before setting the tail to the newly added node.
    else {
      tail->next = newNode;
      tail = newNode;
    }
  }

  void removeNode(int removal_index){
    Node *current_node = head;
    Node *prev = NULL;
    Node *next = NULL;
    bool deletion_complete = false;
    int i = 0;
    while((current_node != NULL) && (deletion_complete == false)){

      if (removal_index == 0){
        head = head->next;
        deletion_complete = true;
      }

      if (i == removal_index-1){
        prev = current_node;
        next = (current_node->next)->next;
        delete current_node->next;
        prev->next = next;
        deletion_complete = true;
      }
      current_node = current_node->next;
      i = i + 1;
      }
    }

  //print linkedlist
  void printLinkedList(){
     Node *current_node = head;

     while (current_node != NULL){
       cout << current_node->data << endl;
       current_node = current_node->next;
     }
  }
};
int main(){

LinkedList ll;
ll.addNode(2);
ll.addNode(3);
ll.addNode(4);
ll.removeNode(0);

ll.printLinkedList();



}
