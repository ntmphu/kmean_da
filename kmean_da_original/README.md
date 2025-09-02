# K-means Domain Adaptation Project

This project implements methods for K-means clustering and domain adaptation using parametric analysis and optimal transport techniques. The goal is to align distributions from different domains while maintaining the integrity of the clustering process.

## Project Structure

- **Kmean_DA_parametric.py**: Contains functions for generating data, computing intervals, and performing parametric analysis related to K-means clustering and domain adaptation.
  
- **OptimalTransport.py**: Includes functions related to optimal transport methods, which are used for aligning distributions in the context of machine learning.
  
- **util.py**: Provides utility functions that are used across the project, such as constructing parameters for quadratic inequalities and computing intersections of intervals.
  
- **intersection.py**: Contains functions for solving and managing interval intersections, which are important for the analysis performed in the project.
  
- **Kmean_DA_oc.py**: Implements overconditioning methods for K-means clustering in the context of domain adaptation.
  
- **Kmean_clustering.py**: Includes functions for performing K-means clustering, including selecting clusters based on certain criteria.

## Usage

1. Ensure you have the required libraries installed (e.g., NumPy, SciPy, Matplotlib).
2. Run the `Kmean_DA_parametric.py` script to perform the analysis.
3. Modify parameters as needed to fit your specific use case.

## Contribution

Feel free to fork the repository and submit pull requests for any improvements or bug fixes.