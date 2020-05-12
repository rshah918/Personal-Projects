#include <iostream>
#include "Product.h"

using namespace std;

int main() {
    Product p(17, "Jill");
    cout << "p (id): " << p.getID() << endl;
    cout << "p (name): " << p.getName() << endl << endl;
    
    Product q(183, "Tom");
    cout << "q (id): " << q.getID() << endl;
    cout << "q (name): " << q.getName() << endl << endl;
    
    Product badP(222, "");
    cout << "badP (id): " << badP.getID() << endl;
    cout << "badP (name): " << badP.getName() << endl << endl;
    
    p.setName("Jill Extreme");
    cout << "p (name): " << p.getName() << endl;
    
    q.setName("");
    cout << "q (name): " << q.getName() << endl;	
    
    p.setDescription("This is a fantastic product.");
    cout << "p (description): " << p.getDescription() << endl;
    
    q.setDescription("");
    cout << "q (description): " << q.getDescription() << endl << endl;

    cout << "Before 1st shipment" << endl;
    cout << "inventory: " << p.getInventoryCount() << endl;
    cout << "totalPaid: " << p.getTotalPaid() << endl;
    cout << "price: " << p.getPrice() << endl;
    cout << "numSold: " << p.getNumberSold() << endl << endl;

    p.addShipment(88, 230.44);

    cout << "After 1st shipment" << endl;
    cout << "inventory: " << p.getInventoryCount() << endl;
    cout << "totalPaid: " << p.getTotalPaid() << endl;
    cout << "price: " << p.getPrice() << endl;
    cout << "numSold: " << p.getNumberSold() << endl << endl;
    
    p.addShipment(133, 142.11);
    
    cout << "After 2nd shipment" << endl;
    cout << "inventory: " << p.getInventoryCount() << endl;
    cout << "totalPaid: " << p.getTotalPaid() << endl;
    cout << "price: " << p.getPrice() << endl;
    cout << "numSold: " << p.getNumberSold() << endl << endl;
    
    cout << "Testing negative quantity" << endl;
    
    if (!p.addShipment(-5, 22.55)) {
        cout << "shipment failed" << endl;
    }
    else {
        cout << "shipment passed" << endl;
    }
    
    cout << "After failed shipment (nothing should change from last output)" << endl;
    cout << "inventory: " << p.getInventoryCount() << endl;
    cout << "totalPaid: " << p.getTotalPaid() << endl;
    cout << "price: " << p.getPrice() << endl;
    cout << "numSold: " << p.getNumberSold() << endl << endl;

    
    cout << "Testing negative shipment cost" << endl;
    
    if (!p.addShipment(5, -22.55)) {
        cout << "shipment failed" << endl;
    }
    else {
        cout << "shipment passed" << endl;
    }
    
    cout << "After failed shipment (nothing should change from last output)" << endl;
    cout << "inventory: " << p.getInventoryCount() << endl;
    cout << "totalPaid: " << p.getTotalPaid() << endl;
    cout << "price: " << p.getPrice() << endl;
    cout << "numSold: " << p.getNumberSold() << endl << endl;
    
    cout << "Testing reduceInventory" << endl;
   
    p.reduceInventory(33);
   
    cout << "After reduceInventory" << endl;
    cout << "inventory: " << p.getInventoryCount() << endl;
    cout << "totalPaid: " << p.getTotalPaid() << endl;
    cout << "price: " << p.getPrice() << endl;
    cout << "numSold: " << p.getNumberSold() << endl << endl;
    
    cout << "Testing reduceInventory (negative quantity)" << endl;   
    if (!p.reduceInventory(-75)) {
        cout << "reduction failed" << endl;
    }
    else {
        cout << "reduction passed" << endl;
    }
   
    cout << "After failed reduceInventory (nothing should change from last output)" << endl;
    cout << "inventory: " << p.getInventoryCount() << endl;
    cout << "totalPaid: " << p.getTotalPaid() << endl;
    cout << "price: " << p.getPrice() << endl;
    cout << "numSold: " << p.getNumberSold() << endl << endl;
  
    cout << "Testing reduceInventory (more than inventory)" << endl;
    if (!p.reduceInventory(500)) {
        cout << "reduction failed" << endl;
    }
    else {
        cout << "reduction passed" << endl;
    }
   
    cout << "After failed reduceInventory (nothing should change from last output)" << endl;
    cout << "inventory: " << p.getInventoryCount() << endl;
    cout << "totalPaid: " << p.getTotalPaid() << endl;
    cout << "price: " << p.getPrice() << endl;
    cout << "numSold: " << p.getNumberSold() << endl << endl;
    
    cout << "More testing should be done, e.g. edge cases etc." << endl;
}