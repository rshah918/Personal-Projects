#include <iostream>
#include <vector>

using namespace std;

vector<int> BubbleSort(vector<int> vec){
  if(vec.size() < 2){//check input size
    return vec;
  }

  bool noSwaps = false;//stop after a pass with no swaps
  while(noSwaps == false){
    int bubbleStart = 0;
    int bubbleEnd = 1;
    bool swapPerformed = false; //if a swap was performed in the current pass
    while(bubbleEnd < vec.size()){
      //compare bubble values
      if(vec.at(bubbleEnd) < vec.at(bubbleStart)){
        //perform swap
        int temp = vec.at(bubbleStart);
        vec[bubbleStart] = vec.at(bubbleEnd);
        vec[bubbleEnd] = temp;
        swapPerformed = true;
      }
      //increment bubble
      bubbleEnd++;
      bubbleStart++;
    }
    //kill outer loop of no swap is performed
    if(swapPerformed == false){
      noSwaps = true;
    }
  }
  return vec;
}

int main(){
  vector<int> vec1;
  vec1.push_back(1000);
  vec1.push_back(200);
  vec1.push_back(30);
  vec1 = BubbleSort(vec1);
  //print vector
  for(int i = 0; i < vec1.size(); i++){
    cout << vec1.at(i) << ',';
  }
  cout << endl;
  return 1;
}
