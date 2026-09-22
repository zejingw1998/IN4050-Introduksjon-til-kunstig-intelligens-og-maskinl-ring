# IN4050 – Introduction to Artificial Intelligence and Machine Learning

This repository contains exercises, implementations, and mandatory assignments from **IN4050 – Introduction to Artificial Intelligence and Machine Learning** at the University of Oslo (UiO), Autumn 2026.

## Repository Structure

* `Fundamental_algorithms/` – Implementations of basic algorithms covered in the course
* `Week_exercises/` – Weekly exercises, notebooks, and small experiments
* `Oblig1/` – Mandatory assignment 1
* `Oblig2/` – Mandatory assignment 2

## Topics

The repository currently contains work related to:

* Search algorithms and state-space search
* Greedy and exhaustive search
* Discrete and continuous optimization
* Travelling Salesman Problem (TSP)
* Supervised learning
* k-Nearest Neighbors (KNN)
* Linear classification
* Decision boundaries
* Model evaluation
* Bias and variance

## Example: Exhaustive Search

One exercise uses exhaustive search to solve a small Travelling Salesman Problem.

```python
for rest in itertools.permutations(range(1, len(cities))):
    candidate = (0,) + rest
    length = tour_length(candidate, cities)

    if length < best_length:
        best_length = length
        best_order = candidate
```

For small problems, every possible route can be evaluated directly and the shortest route selected. This also illustrates how quickly exhaustive search becomes expensive as the number of cities increases.

## Machine Learning Exercises

The machine learning part of the course includes classification tasks such as:

* representing observations with feature vectors
* computing distances between samples
* implementing and testing KNN
* visualizing decision boundaries
* evaluating classification performance

## Tools

Most of the code is written in Python using:

* NumPy
* Matplotlib
* scikit-learn
* Jupyter Notebook
* itertools

## About

I use this repository to keep track of my work in IN4050 throughout the semester. The implementations are mainly written for learning and experimentation, so some of them intentionally use simple approaches before moving to more efficient methods.
