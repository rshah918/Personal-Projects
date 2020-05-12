#ifndef STORE_H
#define STORE_H

#include "Product.h"
#include "Customer.h"

class Store {
	char name[100];
	Product* products[100];
	Customer* customers[100];
    static int counter; // declares but does not define
    int numProducts;
    int numCustomers;
public:
    Store();
    Store(const char name[]);
    const char* getName();
    void setName(const char storeName[]);
    bool addProduct(int productID, const char productName[]);
    Product* getProduct(int productID);
    // even though getProduct should be private, it helps autograding to make it public
    bool addCustomer(int customerID, const char customerName[], bool credit=false);
    Customer* getCustomer(int customerID);
    // even though getCustomer should be private, it helps autograding to make it public
    bool takeShipment(int productID, int quantity, double cost);
    bool sellProduct(int customerID, int productID, int quantity);
    bool takePayment(int customerID, double amount);
    void outputProducts(std::ostream& os);
    void outputCustomers(std::ostream& os);
};

#endif
