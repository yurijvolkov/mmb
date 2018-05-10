# Matrix multiplication benchmark (MMB)

In current research 3 algorithms were implemented: 
* Straight way. Which take O(N^3) ;
* Winograd. Which also take O(N^3), but do less multiplications ;
* Strassen. Which take ~O(N^2.8) ;

Experiment was done for square matrices (NxN), but all methods are implemented that they can multiply non-square matrices.
N was evaluated in [20; 1000) with step 20. Time of execution was stored. All results are accessible in `reports.csv` file.

Main plot (dynamic version located in main_plot.html): ![here](https://raw.githubusercontent.com/yurijvolkov/mmb/master/main_plot.png).

Plot for Strassen algo is "discrete" because this algorithm as input requires square matrices with dimensions 2^K. 
So matrices with shapes non-equal to 2^K are being wrapped to the nearest power of 2. (e.g. 56x117 becomes (128x128)).
