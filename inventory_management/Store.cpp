#include <iostream>
#include "Store.h"
#include "Product.h"
#include "Customer.h"
using namespace std;

int Store::counter = 0; // defines and initializes

Store::Store() : numProducts(0),numCustomers(0), products(), customers() {
  setName("");
}

Store::Store(const char name[]) : numProducts(0),numCustomers(0), products(),customers() {
  setName(name);
}

const char* Store::getName(){
  return name;
}

void Store::setName(const char storeName[]){
  if (strlen(storeName) > 0) {
      strcpy(this->name, storeName);
  }
  else {
      counter++;
      ostringstream oss;
      oss << "Store " << counter;
      strcpy(this->name, oss.str().c_str());
  }
}

bool Store::addProduct(int productID, const char productName[]){
  if(products[0] == NULL){
    products[0] = new Product(productID, productName);
    return true;
  }
  for(int i = 0;i < 100;++i){
    if(products[i] == NULL){
      products[i] = new Product(productID,productName);
      return true;
    }
    if(products[i]->getID()==productID){
      return false;
    }
  }
  return false;
}

Product* Store::getProduct(int productID){
  for(int i = 0;i < 100;++i){
    if(products[i] == NULL){
      return nullptr;
    }
    if (products[i]->getID() == productID){
      return products[i];
    }
  }
  return nullptr;
}

bool Store::addCustomer(int customerID, const char customerName[], bool credit){
  if(customers[0] == NULL){
    customers[0] = new Customer(customerID, customerName,credit);
    return true;
  }
  for(int i = 0;i < 100;++i){
    if(customers[i] == NULL){
      customers[i] = new Customer(customerID, customerName, credit);
      return true;
    }
    if(customers[i]->getID()==customerID){
      return false;
    }
  }
  return false;
  }

Customer* Store::getCustomer(int customerID){
  for(int i = 0; i<100; ++i){
    if (customers[i] == NULL){
      return nullptr;
    }
    if(customerID == customers[i]->getID()){
      return customers[i];
    }
  }
  return NULL;
}

bool Store::takeShipment(int productID, int quantity, double cost){
  
  if((quantity < 0) || (cost < 0)){
    return false;
  }
  for(int i = 0;i < 100;++i){
    if(products[i] == NULL){
      return false;
    }
    if(products[i]->getID() == productID){
      products[i]->addShipment(quantity, cost);
      return true;
    }
  }
  return false;
}

bool Store::sellProduct(int customerID, int productID, int quantity){

  //loop thorugh customers and products
    //find indec of customer that matches customer ID (same for products)
    //using that index, check to see for customers if they have credit
    //for products, if there is enough quantity
    //products[index]->getInventoryCount()<quantity;
      //if the above is true, then return false
int customerindex = 0;
int productIndex = 0;
  for(int i = 0;i < 100;++i){
    if(products[i] == NULL){
      return false;
    }

    if(products[i]->getID() == productID){
      productIndex = i;
      break;
    }
  }
  for(int i = 0;i < 100;++i){
    if(customers[i] == NULL){
      return false;
    }
    if(customers[i]->getID() == customerID){
      customerindex = i;
      break;
    }
  }
if(quantity < 0){
  return false;
}

if(products[productIndex]->getInventoryCount() < quantity){
  return false;
}


  int amount = quantity * products[productIndex]->getPrice();
  customers[customerindex]->processPurchase(amount, *products[productIndex]);
  products[productIndex]->reduceInventory(quantity);
  return true;
  }

bool Store::takePayment(int customerID, double amount){
  if(amount < 0){
    return false;
  }
  for(int i = 0;i < 100;++i){
    if(customers[i] == NULL){
      return false;
    }
    if(customers[i]->getID() == customerID){
      customers[i]->processPayment(amount);
      return true;
    }
  }
  return false;
}

void Store::outputProducts(std::ostream& os){
  for(int i= 0;i < 100;++i){
    if(products[i] != NULL){
      os << *products[i] << endl;
    }
  }
}

void Store::outputCustomers(std::ostream& os){
  for(int i= 0;i < 100;++i){
    if(customers[i] != NULL){
      os << *customers[i] << endl;
    }
  }
}
