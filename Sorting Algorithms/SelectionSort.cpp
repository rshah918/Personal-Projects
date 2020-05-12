#include <iostream>
#include <vector>

using namespace std;

//Selection Sort: find mimimum, append to front

vector<int> SelectionSort(vector<int> vec){

  int boundary = 0;
  while(boundary < vec.size()){
    int min = boundary;
    //find min in vec[boundary:-1]
    for(int i = boundary; i < vec.size(); i++){
      if(vec.at(i) < vec.at(min)){
        min = i;
      }
    }
    //increment boundary, add minimum to the end of the sorted section
    swap(vec[boundary], vec[min]);
    boundary++;
  }
  return vec;
}


int main(){
  vector<int> nums;
  nums.push_back(1);
  nums.push_back(3);
  nums.push_back(274);
  nums.push_back(2000);
  nums.push_back(200);
  nums.push_back(2);
  nums = SelectionSort(nums);
  for(int i = 0; i < nums.size(); i++){
    cout << nums.at(i);
    if(i != nums.size()-1){
      cout << ',';
    }
  }
  cout << endl;
}
