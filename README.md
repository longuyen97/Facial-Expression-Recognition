# Facial Expression Recognition

Facial expressions are a form of nonverbal communication between human beings. While we can choose our words very carefully 
to avoid involuntary emotions revealing, it could be difficult to control our facial expressions. 

For a long time (and still), it is considered very difficult for machines to classify the emotions displayed on human faces. Many
approaches were tested (with and without significant success) to extract the facial expression and gather them into categories. A successful
automation of this problem could lead to many applications in human-computer interaction to improve the fluency, accuracy and naturalness 
of the interactions.

This repository is an attempt to employ Support Vector Machines and Principal Component Analysis (both come from the field of 
Statistical Machine Learning) to filter unimportant features and finding important patterns in RGB images of human faces.

## Datasets

[CKPLUS](https://www.kaggle.com/shawon10/ckplus) is a lite version of Google's [facial expression comparison dataset](https://research.google/tools/datasets/google-facial-expression/).

This dataset is a large-scale facial expression dataset consisting of face image triplets along with human annotations that specify which two faces in each triplet 
form the most similar pair in terms of facial expression. Each triplet in this dataset was annotated by six or more human raters. This dataset is intended to aid researchers working on 
topics related to facial expression analysis such as expression-based image retrieval, expression-based photo album summarization, emotion classification, expression synthesis, etc.

Following are some samples of the dataset

<table style="width:100%">
  <tr>
    <td><img src="data/anger/S022_005_00000032.png"></td>
    <td><img src="data/disgust/S005_001_00000010.png"></td>
    <td><img src="data/fear/S011_003_00000013.png"></td>
    <td><img src="data/happy/S010_006_00000014.png"></td>
    <td><img src="data/sadness/S011_002_00000021.png"></td>
    <td><img src="data/surprise/S010_002_00000013.png"></td>
  </tr>
  <tr>
    <td>anger</td>
    <td>disgust</td>
    <td>fear</td>
    <td>happy</td>
    <td>sadness</td>
    <td>surprise</td>
  </tr>
</table>

## Implementation details

Assuming we want to classify images between two targets `happy` and `anger`. Assigning images of `happy`with target 1 as well 
as assigning images of `anger` with target -1. Each pixel of a `48x48` images will be an attribute. An image therefore 
will be a vector `xi` in a vector space `R^2304`. The task of the algorithm is assigning `xi` with either -1 or 1.  

The main ideas of the following concepts come from the course [Statistical Machine Learning](https://www.youtube.com/playlist?list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC). 
An understanding of the mathematical definitions is NOT necessary to understand the algorithm, but it helps the reasoning process of decision and classification.

#### Hard Margin Classifier 

The course [Statistical Machine Learning](https://www.youtube.com/watch?v=0cZwSzsE-UA&list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC&index=18) of University Tübingen 
does a really good job explain the naive classifier variation. The idea of Hard Margin Classifier is to search for the 
optimal hyperplane with the largest distance to both classes. The ideal hyperplane in this case has the same distance between
all two classes; the greatest distance gives the algorithm its robustness against outliers.   

<div align="center">
<img src="data/hard-margins.png" height="400px" align="center">
</div>

Assuming the data can be perfectly separated by an ideal hyperplane, then the classifier can take the linear form

```
f(x) = sign(<w, x> + b)
```

The canonical form of the hyperplane consequently takes the form

```
<w, x> + b = 0
```

The margin of an arbitrary hyperplane H is defined as the minimal distance of all training points to the hyperplane.

```
p(H, x1,..., xn) = min{1,..,n} d(xi, H)
```

Therefore, the formulation of the hard margin optimization problem is to maximize the margin (assuming all points 
are on the correct side of the hyperplane and outside of the margin) or 

```
minimize{w, b} 1/2 ||w|| ^ 2
subject to Yi(<w, Xi> + b) >= 1 for all i = 1,...,n
``` 

#### Soft Margin Classifier 

The hard margin may be too strict for data that is not linear separable. In this case we want to allow the separating
hyperplane to make some errors (sacrifice a low bias for a low variance). 

<div align="center">
<img src="data/hard-and-soft-margins.png" height="400px" align="center">
</div>

The optimization problem for this hyperplane can be formulated as following where `C` is a constant that controls the tradeoff between two terms:

```
minimize{x, b, slack} 1 / 2 ||w|| ^ 2 + C/n sum{i} slack{i}
subject to Yi(<w, Xi> + b) >= 1 - slack{i} for all i = 1,..., n 
```

The slack variable is supposed to measure how far a variable is sitting on the wrong side 
of the hyperplane and will be used as a regularisation term of the optimization problem. A good hyperplane is where the 
sum of the punishments is as small as possible.  

#### Polynomial kernel for non-linear modelling

Soft margin classifier is an initial step into the right direction, however the linear functions are general still too 
restrictive despite simple conceptual appeal. In reality this lead normally to underfitting the dataset and missing out 
very much potential of performance. 

In order to overcome this problem, we could use a feature map with basis functions to represent more complex functions, say
polynomials. But:
- It is not so obvious which are good basis functions
- We need to fix the basis before we see the data. This means that we need to have very many basis functions to be flexible, this 
leads to a very high dimensional representations of our data. 

#### Principal Component Analysis

## References
- [Google facial expression comparison dataset](https://research.google/tools/datasets/google-facial-expression/)
- [CKPLUS](https://www.kaggle.com/shawon10/ckplus)
- [Wikipedia - Support vector machines](https://de.wikipedia.org/wiki/Support_Vector_Machine)
- [Uni Tübingen - Statistical Machine Learning Part 16 - Support vector machines: hard and soft margin](https://www.youtube.com/watch?v=0cZwSzsE-UA&list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC&index=18)
- [Uni Tübingen - Statistical Machine Learning Part 18 - Kernels: definitions and examples](https://www.youtube.com/watch?v=ABOEE3ThPGQ&list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC&index=20)
- [StatQuest: Support Vector Machines, Clearly Explained!!!](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [StatQuest: Support Vector Machines: The Polynomial Kernel](https://www.youtube.com/watch?v=Toet3EiSFcM)
- [StatQuest: PCA main ideas](https://www.youtube.com/watch?v=HMOI_lkzW08)