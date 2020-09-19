# Facial Expression Recognition

### Datasets

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

### Implementation details

Assuming we want to classify images between two targets `happy` and `anger`. Assigning images of `happy`with target 1 as well 
as assigning images of `anger` with target -1. Each pixel of a `48x48` images will be an attribute. An image therefore 
will be a vector `xi` in a vector space `R^2304`. The task of the algorithm is assigning `xi` with either -1 or 1.  

##### Hard Margin Classifier 

The course [Statistical Machine Learning](https://www.youtube.com/watch?v=0cZwSzsE-UA&list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC&index=18) of University Tübingen 
does a really good job explain the naive classifier variation. The idea of Hard Margin Classifier is to search for the 
optimal hyperplane with the largest distance to both classes. The ideal hyperplane in this case has the same distance between
all two classes; the greatest distance gives the algorithm its robustness against outliers.   

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

##### Soft Margin Classifier 

The hard margin may be too strict for data that is not linear separable. In this case we want to allow the separating
hyperplane to make some errors (sacrifice a low bias for a low variance). The optimization problem for this hyperplane 
can be formulated as:

```
minimize{x, b, slack} 1 / 2 ||w|| ^ 2 + C/n sum{i} slack{i}
subject to Yi(<w, Xi> + b) >= 1 - slack{i} for all i = 1,..., n 
```

### Features reduction

##### K-Mean clustering

##### Principal Component Analysis

### References
- [Google facial expression comparison dataset](https://research.google/tools/datasets/google-facial-expression/)
- [CKPLUS](https://www.kaggle.com/shawon10/ckplus)
- [Wikipedia - Support vector machines](https://de.wikipedia.org/wiki/Support_Vector_Machine)
- [Uni Tübingen - Statistical Machine Learning Part 16 - Support vector machines: hard and soft margin](https://www.youtube.com/watch?v=0cZwSzsE-UA&list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC&index=18)
- [StatQuest: Support Vector Machines, Clearly Explained!!!](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [StatQuest: Support Vector Machines: The Polynomial Kernel](https://www.youtube.com/watch?v=Toet3EiSFcM)
- [StatQuest: PCA main ideas](https://www.youtube.com/watch?v=HMOI_lkzW08)