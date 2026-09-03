
# IN4050 – Introduction to Artificial Intelligence and Machine Learning

This repository contains my exercises, notes, and code for **IN4050 – Introduction to Artificial Intelligence and Machine Learning** at the University of Oslo (UiO).

## Course Information

- **Course code:** IN4050
- **Course name:** Introduction to Artificial Intelligence and Machine Learning
- **University:** University of Oslo (UiO)
- **Semester:** Autumn 2026
- **Field:** Artificial Intelligence and Machine Learning

## Repository Structure

The repository is organized mainly by weekly exercises.
- `Oblig/` Mandatory assignment, project
- `Week_exercises/` – Weekly exercises, notebooks, code, and notes
- `README.md` – Repository description

## Topics Covered

Topics covered in this repository may include:

- Artificial Intelligence fundamentals
- Search algorithms
- State-space search
- Greedy search
- Exhaustive search
- Optimization
- Discrete optimization
- Continuous optimization
- Travelling Salesman Problem (TSP)
- Supervised learning
- Classification
- k-Nearest Neighbors (KNN)
- Linear models
- Decision boundaries
- Model evaluation
- Bias and variance
- Machine learning fundamentals

## Programming and Tools

The exercises are mainly implemented using:

- Python
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook
- itertools

## Example: Exhaustive Search

One of the exercises explores exhaustive search for a small Travelling Salesman Problem (TSP).

The basic idea is to evaluate every possible route and keep track of the shortest one found.

    for rest in itertools.permutations(range(1, len(cities))):
        candidate = (0,) + rest
        length = tour_length(candidate, cities)

        if length < best_length:
            best_length = length
            best_order = candidate

This is a simple example of discrete optimization, where the possible solutions are different permutations of the cities.

## Classification

The course also introduces supervised learning methods for classification problems, including k-Nearest Neighbors.

Typical tasks include:

- Representing data using feature vectors
- Calculating distances between observations
- Finding nearest neighbors
- Predicting class labels
- Visualizing decision boundaries
- Evaluating classification performance

## Learning Goals

The main purpose of this repository is to document my progress throughout IN4050 and organize the exercises completed during the semester.

Through the course and exercises, I aim to develop a better understanding of:

- How search algorithms explore solution spaces
- How optimization problems are formulated
- How discrete and continuous optimization differ
- How machine learning models learn from data
- How classification algorithms work
- How mathematical concepts can be implemented in Python
- How to analyze and evaluate machine learning models

## Notes

This repository is primarily intended for educational purposes.

The code reflects my own learning process during the course. Some implementations may be simplified for educational purposes and may be updated or improved throughout the semester.

## Author

**Zejing Wang**

Master's student in Mathematics for Applications  
University of Oslo (UiO)

## Disclaimer

This repository contains personal course notes and exercise solutions.

If you are currently taking IN4050, please follow the University of Oslo's rules regarding collaboration and academic integrity.
