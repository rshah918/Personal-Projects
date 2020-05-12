#include <iostream>

using namespace std;

class Node{

  public:
    int data;
    Node* next;
  public:
    Node(int val){
      data = val;
      next = NULL;
    }
};
class Queue{
  private:
    Node* head;
    Node* tail;
  public:
    Queue(){
      head = NULL;
      tail = NULL;
    }
    void Enqueue(int val){
      Node* newNode = new Node(val);
      if(head == NULL){
        head = newNode;
        tail = newNode;
      }
      else{
        tail->next = newNode;
        tail = newNode;
      }
    }
    void Dequeue(){
      if(head == NULL){
        cout << "Queue is empty!" << endl;
      }
      else{
        cout << head->data << endl;
        head = head->next;
      }
    }
};

int main(){
  Queue q1 = Queue();
  q1.Enqueue(1);
  q1.Enqueue(2);
  q1.Enqueue(3);
  q1.Enqueue(4);
  q1.Dequeue();
  q1.Dequeue();
  q1.Dequeue();
  q1.Dequeue();
  q1.Dequeue();
}
