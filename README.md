## Tensor format

The input format is expected to start with the number of dimension of the tensor followed by the length of each dimension in the next line. The following lines will have the coordinates and values of each nonzero elements.

An example of a 3x3x3 tensor - toy.tns:  
3  
3 3 3  
1 1 1 1.00  
1 2 2 2.00  
1 3 1 10.00  
2 1 3 7.00    
2 3 1 6.00    
2 3 2 5.00  
3 1 3 3.00  
3 2 2 11.00   

## Build 

$ make  

## Run

Example:

$ ./mttkrp -i toy.tns -m 0 -R 32 -t 1  

To see all the options: 

$ ./mttkrp --help

Avaiable options:
-t 1: COO on CPU  
-t 2: CSF on CPU  
-t 3: COO on GPU  
-t 4: CSF on GPU  
-t 8: B-CSF on GPU  
-t 10: HB-CSF on GPU  

