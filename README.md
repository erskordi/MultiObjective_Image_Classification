# MultiObjective_Image_Classification

The present repo utilizes a multi-objective optimization algorithm where multiple pre-trained AI models are assessed given their evaluations on the test data.

The current version is based on the Fashion MNIST dataset, but more datasets will be used soon. 

## Objectives

The objectives used currently are:
- AUC
- Memory utilization
- FLOPS
- Energy consumption

Assessment occurs in two stages:
1. Multi-objective (Pareto front)
2. Each objective separately, subject to constraints