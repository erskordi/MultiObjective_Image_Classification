# MultiObjective_Image_Classification

The present repo utilizes a multi-objective optimization algorithm where multiple pre-trained AI models are assessed given their evaluations on the test data.

Currently, we provide results from three datasets:
- `FashionMNIST`
- `CIFAR10`
- `Gravity Spy (Gravitational Waves)` (source: [!Kaggle](https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves))

## Models

We use three deep neural network architectures in this project:
- `Vision Transformers (ViT)`: state-of-the-art models in digital image processing, but are usually very large, with millions of hyperparameters. They function by identifying similarities between features in an image, leading to very robust classification and image generation abilities.
- `Convolutional Neural Nets (CNN)`: The workhorse of digital image processing before the emergence of the ViT. They use convolutions to extract image features, before feeding the features in a regular feed-forward network to perform classification. They are also used on image generation applications (e.g., U-Net).
- `Parallel Graph Neural Networks (PGNN)`: Implement parallel graph neural networks. PGNN first extracts key spatial-spectral image features using PPCA, followed by preprocessing to obtain two primary features and a normalized adjacency matrix. A parallel architecture is constructed using improved GCN and ChebNet to extract local and global spatial-spectral features, respectively. Finally, the discriminative features obtained through the fusion strategy are input into the classifier to obtain the classification results.

## Objectives

The objectives used in our multi-objective approach are:
- Area Under Curve
- Memory utilization (in MB)
- FLOPS
- Energy consumption (in J)
- Number of parameters / model

Compared to AUC, the rest of the metrics are orders of magnitude larger. Thus, we report their logarithmic values.

The framework assessment is split into two paths:
1. Multi-objective optimization (Pareto front)
    - This branch incorporates all the objectives as: $\min(\text{Memory Utilization}, \text{FLOPS}, \text{Energy Consumption}, \text{Number of parameters}, -\text{AUC})$
    - Due to the objective being 5D, we opted for plotting the Pareto front by flattening the number of parameters and AUC dimensions. Each point corresponds to representatives of different families of architectures.
2. Single objective optimization. Each objective is evaluated separately while being subjected to constraints imposed by the models themselves. For example:
    - $\min(FLOPS)$ subject to: 
        * $\text{Energy Consumption}\leq\text{Maximum Energy Consumption}$
        * $\text{Memory Utilization}\leq\text{Maximum Memory Utilization}$
        * $\text{Model parameters}\leq\text{Maximum Model parameters}$
        * $\text{AUC}\geq\text{Minimum AUC}$
    - Rest of the objectives are similarly achieved.

## Visualization

For each dataset, we provide the following plottings:
- Pareto front (only for the multi-objective approach).
- Pairwise plots between constraints and each objective. These show potential patterns between models of the same architecture family.