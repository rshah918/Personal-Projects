#include <iostream>
using namespace std;

bool binarySearch(int arr[], int left, int right, int target){
  if(left <= right){
    int middle = ((right-left)/2) + left;
    if(arr[middle] < target){
      bs(arr, middle + 1, right, target);
    }
    else if(arr[middle] > target){
      bs(arr, left, middle - 1, target);
    }
    else{
      cout << "Found" << endl;
      return true;
    }
  }
  else{
    cout << "Not Found!" << endl;
    return false;
  }
}

int main(){
  int arr[12] = {1,2,3,4,7,9,13,56,79,88,89,90};
  binarySearch(arr,0,11,90);
  return 1;
}
