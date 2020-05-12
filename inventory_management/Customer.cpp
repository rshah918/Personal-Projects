#include <cstring>
#include <sstream>
#include <iostream>
#include "Customer.h"
#include "Product.h"

using namespace std;

int Customer::counter = 0; // defines and initializes



Customer::Customer(int customerID, const char name[], bool hasCredit) :
id(customerID), balance(0.0), numPurchased(0), credit(hasCredit),productsPurchased() {
  setName(name);
}

int Customer::getID() const { return id; }

const char* Customer::getName() const {
    return name;
}

void Customer::setName(const char customerName[]){
  if (strlen(customerName) > 0) {
      strcpy(this->name, customerName);
  }
  else {
      counter++;
      ostringstream oss;
      oss << "Customer " << counter;
      strcpy(this->name, oss.str().c_str());
  }
}

bool Customer::getCredit() const { return credit; }

void Customer::setCredit(bool hasCredit) {
    credit = hasCredit;
}

double Customer::getBalance() const { return balance; }

bool Customer::processPayment(double amount){
  if((amount) < 0){
    return false;
  }
  else{
    balance = (balance + amount);
    return true;
  }
}

bool Customer::processPurchase(double amount, Product product){
  if(credit == true){
    balance = (balance - amount);
  }
  else{
    if(balance >= amount){
      balance = (balance - amount);
      return true;
    }
    else{
      return false;
    }
  }
  if(amount < 0){
    return false;
  }
  for(int i = 4; i > 0; --i){

    strcpy(productsPurchased[i], productsPurchased[i-1]);
  }
  strcpy(productsPurchased[0], product.getName());
  return true;
}
void Customer::outputRecentPurchases(std::ostream& os) const{
   os << "Products Purchased --\n";

   for(int i = 0; i < 5; ++i){
     if(productsPurchased[i][0] != '\0'){
       os << productsPurchased[i] << endl;
     }
   }

  }
  std::ostream& operator<<(std::ostream& os, const Customer& customer){
    os << "Customer Name: " << customer.getName() << endl;
    os << "Customer ID: " << customer.getID() << endl;
    os << std::boolalpha;
    os << "Has Credit: " << customer.getCredit() << endl;
    os << "Balance: " << customer.getBalance() << endl;
    customer.outputRecentPurchases(os);
    return os;
}
