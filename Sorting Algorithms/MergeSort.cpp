#include <iostream>
#include <vector>

using namespace std;
void printvec(vector<int> vec){
  for(int i = 0; i < vec.size(); i++){
    cout << vec.at(i);
    if(i != vec.size()-1){
      cout << ',';
    }
  }
  cout << endl;
}

void merge(vector<int> *vec1, int left, int middle, int right){
  //split into left and right subarrays
  int size1 = middle-left+1;
  int size2 = right-middle;

  vector<int> leftVec;
  vector<int> rightVec;
  //populate sub vectors
  for(int i = 0; i < size1; i++){
    leftVec.push_back(vec1->at(left + i));
  }
  for(int j = 0; j < size2; j++){
    rightVec.push_back(vec1->at(j + middle+1));
  }

  //merge
  int i = 0;
  int j = 0;
  int k = left;

  while((i < size1) && (j < size2)){
    if(leftVec.at(i) < rightVec.at(j)){
      vec1->at(k) = (leftVec.at(i));
      i++;
    }
    else{
      vec1->at(k) = (rightVec.at(j));
      j++;
    }
    k++;
  }
  //add remaining elements
  while(i < size1){
    vec1->at(k) = (leftVec.at(i));
    i++;
    k++;
  }
  while(j < size2){
    vec1->at(k) = (rightVec.at(j));
    j++;
    k++;
  }
}

void mergeSort(vector<int> *vec1, int left, int right){
  if(left < right){
    int middle = (left+right)/2;
    mergeSort(vec1, left, middle);
    mergeSort(vec1, middle+1, right);
    merge(vec1, left, middle, right);
  }
}
int main(){
  vector<int> vec;
  vec.push_back(2);
  vec.push_back(1);
  vec.push_back(5);
  vec.push_back(24);
  vec.push_back(3);
  mergeSort(&vec, 0, vec.size()-1);
  printvec(vec);
  return 1;
}
