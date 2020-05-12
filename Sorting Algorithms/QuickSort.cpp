#include <iostream>
#include <vector>
using namespace std;
//For quicksort:
  //i starts at -1, j starts at 0
  //if arr[j] < pivot, increment i and swap arr[i] and arr[j]
  //after iterating, swap arr[i+1] and arr[pivot]
  //return i+1
  //this is to prevent the case where you end up 1 swap short for some inputs
int Partition(int arr[], int start, int end){
  //set pivot to last element
  int pivot = arr[end];
  int partitionIndex = start-1;
  for(int j = start; j < end; j++){
    //increment both partitionIndex and j only if arr[partitionIndex] < arr[pivot]
    if(arr[j] <= pivot){
      partitionIndex++;
      //swap i and j
      swap(arr[partitionIndex], arr[j]);
    }
  }
    //swap i+1 and pivot
  swap(arr[partitionIndex+1], arr[end]);
  return partitionIndex+1;
}


void QuickSort(int arr[], int start, int end){
  //call quicksort on subarrays on both sides of the pivot
  if(start < end){
    int partitionIndex = Partition(arr, start, end);
    QuickSort(arr, start, partitionIndex-1);
    QuickSort(arr, partitionIndex+1, end);
  }
}

int main(){
  int arr[6] = {1,3,274,2000,200,2};
  QuickSort(arr, 0, 5);
  //print array
  for(int i = 0; i < 6; i++){
    cout << arr[i] << endl;;
  }
  return 1;
}
