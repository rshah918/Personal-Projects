#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

class Node{
  public:
    int data;
    Node* next;
    Node(int value){
      data = value;
      next = NULL;
    }
};


class heap{
  vector<int> heapvec;
  public:
    heap(vector<int> arr){
      heapvec = arr;
    }

    void printHeap(){
      for(int i = 0; i < heapvec.size(); i++){
        cout << heapvec.at(i);
        if( i != heapvec.size() - 1){
          cout << '-';
        }
      }
      cout << endl;
    }
    //"sift up".
    void heapify(){
      //get index of last parent
      int lastParent = (heapvec.size()/2.0)-1;
      if(lastParent < 0){
        lastParent = 0;
      }

      for(int i = lastParent; i >= 0; i--){
        //set leftChild and rightChild, with boundary checking
        int rightChild;
        int leftChild = (i*2) + 1;
        if((i*2) + 2 < heapvec.size()){
          rightChild = (i*2) + 2;
        }
        //if the last non leaf node has no right child, then out of range
        //set = to leftChild
        else{
          rightChild = leftChild;
        }
        //maxheap condition
        if(heapvec.at(i) < max(heapvec.at(leftChild), heapvec.at(rightChild))){
          //get greatest child. max() returns a const so do manually
          if (heapvec.at(leftChild) >= heapvec.at(rightChild)){
            //perform swap
            int temp = heapvec.at(leftChild);
            heapvec[leftChild] = heapvec.at(i);
            heapvec[i] = temp;
          }
          else{//rightChild > leftChild
            //perform swap
            int temp = heapvec.at(rightChild);
            heapvec[rightChild] = heapvec.at(i);
            heapvec[i] = temp;
          }
        }
      }
    }
    void siftDown(){
      //edge case
      if(heapvec.size() == 0){
        return;
      }

      while(1){
        int root = 0;
        int leftChild = root*2 + 1;
        int rightChild = root*2 + 2;
        //boundary checking
        if(leftChild > heapvec.size() - 1){
          break;
        }
        //boundary checking
        if(rightChild > heapvec.size()-1){
          rightChild = leftChild;
        }
        if(heapvec.at(root) < max(heapvec.at(leftChild), heapvec.at(rightChild))){
          //get greatest child. max() returns a const so do manually
          if (heapvec.at(leftChild) >= heapvec.at(rightChild)){
            //perform swap
            swap(heapvec[root], heapvec[leftChild]);
          }
          else{//rightChild > leftChild
            //perform swap
            swap(heapvec[root], heapvec[rightChild]);
          }
        }
        //vector is heapified
        else{
          break;;
        }
      }
    }
    void insert(int val){
      heapvec.push_back(val);
      heapify();
      printHeap();
    }
    void getMax(){
      while(heapvec.size() > 0){
        cout << "Top Node before Removal: " << heapvec.at(0) << endl;
        heapvec[0] = heapvec.at(heapvec.size()-1);
        heapvec.pop_back();
        siftDown();
        printHeap();
      }
    }

};

int main(){
  vector<int> vec1;
  vec1.push_back(1);
  vec1.push_back(2);
  vec1.push_back(4);
  heap heap1 = heap(vec1);
  heap1.insert(3);
  heap1.getMax();



  return 1;
}
