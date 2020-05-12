#include <cstring>
#include <sstream>
#include "Product.h"

using namespace std;

int Product::counter = 0; // defines and initializes

Product::Product(int productID, const char productName[]) :
        id(productID), inventory(0), numSold(0), totalPaid(0.0), name(""), description("") {
    setName(productName);
}

int Product::getID() const { return id; }

const char* Product::getName() const {
    return name;
}

void Product::setName(const char productName[]) {
    if (strlen(productName) > 0) {
        strcpy(this->name, productName);
    }
    else {
        counter++;
        ostringstream oss;
        oss << "Product " << counter;
        strcpy(this->name, oss.str().c_str());
    }
}

const char* Product::getDescription() const { return description; }

void Product::setDescription(const char description[]) {
    strcpy(this->description, description);
}

int Product::getNumberSold() const { return numSold; }

double Product::getTotalPaid() const { return totalPaid; }

int Product::getInventoryCount() const { return inventory; }

bool Product::addShipment(int shipmentQuantity, double shipmentCost) {
    if (shipmentQuantity < 0 || shipmentCost < 0) {
        return false;
    }
    else {
        inventory += shipmentQuantity;
        totalPaid += shipmentCost;
    }
    return true;
}

bool Product::reduceInventory(int purchaseQuantity) {
    if (purchaseQuantity < 0)
        return false;
    if (inventory < purchaseQuantity)
        return false;
    inventory -= purchaseQuantity;
    numSold += purchaseQuantity;
    return true;
}

double Product::getPrice() const {
    if ((inventory + numSold) == 0)
        return -1;
    return 1.25 * totalPaid / (inventory + numSold);
}

 std::ostream& operator<<(std::ostream& os, const Product& product) {
     os << "Product Name: " << product.getName() << endl;
     os << "Product ID: " << product.getID() << endl;
     os << "Description: " << product.getDescription() << endl;
     os << "Inventory: " << product.getInventoryCount() << endl;
     os << "Number Sold: " << product.getNumberSold() << endl;
     os << "Total Paid: " << product.getTotalPaid() << endl;
     os << "Price: " << product.getPrice() << endl;
     return os;
 }
