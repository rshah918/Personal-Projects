#include <iostream>

using namespace std;

class Node{
  public:
    int data;
    Node * leftChild;
    Node * rightChild;

    Node(int value){
      data = value;
      leftChild = NULL;
      rightChild = NULL;
    }
};

class bst{
  public:
    Node * root;
    bst(int val){
      Node * newNode = new Node(val);
      root = newNode;
    }
    
    void insert(int val){
      //create a new Node
      Node * newNode = new Node(val);
      //traverse through the bst
      Node * curr = root;
      while(1){
        if(curr->data >= val){
          //if leftchild empty, insert and break;
          if(curr->leftChild == NULL){
            curr->leftChild = newNode;
            break;
          }
          //traverse left
          else{
            curr = curr->leftChild;
          }
        }
        else if(curr->data < val){
          //if rightchild empty, insert and break;
          if(curr->rightChild == NULL){
            curr->rightChild = newNode;
            break;
          }
          //traverse right
          else{
            curr = curr->rightChild;
          }
        }
      }
    }
    Node * search(int val){
      Node * curr = root;
      while(curr != NULL){
        if(curr->data < val){
          curr = curr->rightChild;
        }
        else if(curr->data > val){
          curr = curr->leftChild;
        }
        else{
          return curr;
        }
      }
      cout << "Number not found!" << endl;
      exit(1);
    }
    Node* getParent(int val){
      Node* prev = root;
      Node * curr = root;
      while(curr != NULL){
        if(curr->data < val){
          prev = curr;
          curr = curr->rightChild;
        }
        else if(curr->data > val){
          prev = curr;
          curr = curr->leftChild;
        }
        else{
          return prev;
        }
      }
      cout << "Number not found!" << endl;
      exit(1);
    }
    Node * getInorderSuccessor(Node* curr){
      while(curr->leftChild != NULL){
        curr = curr->leftChild;
      }
      return curr;
    }
    void deleteNode(int val){
      Node * curr = search(val);
      Node* parent = getParent(val);
      //deleting root
      if(curr == root){
        delete curr;
        root = NULL;
      }
      //if node to delete is a leaf
      else if(curr->leftChild == NULL && curr->rightChild == NULL){
        if(parent->leftChild == curr){
          parent->leftChild = NULL;
          delete(curr);
        }
        else{
          parent->rightChild = NULL;
          delete(curr);
        }
      }
      //if node has 1 Child
      else if((curr->leftChild == NULL && curr->rightChild != NULL) | (curr->leftChild != NULL && curr->rightChild == NULL)){
        //swap parent with child
        if(curr->leftChild == NULL){
          curr->data = curr->rightChild->data;
        }
        else{
          curr->data = curr->leftChild->data;
        }
        delete(curr);
      }
      //if the node has 2 children
      //replace node to delete with the smallest node in curr's right subtree
      else if((curr->leftChild != NULL) && (curr->rightChild != NULL)){
        Node* ioSuccessor = getInorderSuccessor(curr);
        int temp = ioSuccessor->data;
        deleteNode(temp);
        curr->data = temp;
      }
    }
  void printTree(Node * root) {
    //print in preorder traversal
    if(root != NULL){
        printf("%d \n", root->data);
        printTree(root->leftChild);
        printTree(root->rightChild);
    }
  }
};

int main(){
  bst bst1 = bst(5);
  bst1.insert(6);
  bst1.insert(2);
  bst1.insert(1);
  bst1.insert(7);
  bst1.deleteNode(7);
  bst1.printTree(bst1.root);
  return 1;
}
