#include <iostream>

using namespace std;

class Node{
  public:
    int data;
    Node* prev;
    int index;
  public:
    Node(){
      prev = NULL;
    }
    Node(int val){
      data = val;
      prev = NULL;
    }
};

class arrayStack{
  bool EmptyIndices[300];
  int tops[3] = {0,0,0};
  Node* stacks = new Node[300];
  /*'''Create an array of nodes. New elements will be added in the first free element in the
   slot list, and will point to the previous node in that stack.'''*/
  public:
    void arrayStack(){
      //populate boolean array with "true" values
      for(int i = 0; i < sizeof(EmptyIndices); i++){
        EmptyIndices[i] = true;
      }
    }
    void push(int stacknum, int val){
      int insertionindex = 301; //301 is an invalid value, will remain as 301 if array is full
      //get the next free index
      for(int i = 0; i < sizeof(EmptyIndices); i++){
        if (EmptyIndices[i] == true){
          insertionindex = i;
          EmptyIndices[i] = false;
          break;
        }
      }
      //error handling
      if(insertionindex == 301){
        cout << "Stacks are full!" << endl;
        return;
      }
      //get the top node
      int TopIndex = tops[stacknum];
      //initialize the new node
      Node* newNode = new Node(val);
      newNode->prev = &stacks[TopIndex];
      newNode->index = insertionindex;
      //push to stack
      stacks[insertionindex] = *newNode;
      //update tops
      tops[stacknum] = insertionindex;
    }
  int pop(int stacknum){
    int TopIndex = tops[stacknum];
    Node topNode = stacks[TopIndex];
    if(EmptyIndices[TopIndex] == true){
      cout << "Stack is empty!!" << endl;
      return 0;
    }
    cout << topNode.data << endl;
    //update EmptyIndices array
    EmptyIndices[topNode.index] = true;
    //get the index of the new stack top
    int newTopIndex = topNode.prev->index;
    tops[stacknum] = topNode.prev->index;
    return 1;
  }

};

int main(){
  arrayStack as1;
  as1.push(0,1);
  as1.push(0,2);
  as1.push(0,3);
  as1.pop(0);
  as1.pop(0);
  as1.pop(0);


}
