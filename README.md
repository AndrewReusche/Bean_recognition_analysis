# Automation Integration Via Machine Learning Dermason Bean Classification

Author: Andrew Reusche

## Business problem: 

A food manufacturer who buys bulk quantities of beans wants to see if classification via machine learning can be utilized to help automate some of their production systems and improve their manufacturing efficiency.

### Test case area of concern:

The manufacturer imports (7) different types of beans that all go through the same clean and wash cycle together. During this cycle, the beans are all mixed together and need to be separated into their respective 7 categories as effectively as possible. The current method of separation is to have the mixture of beans go down a single conveyor where teams of workers pick out the respective different kinds of beans by hand, and deliver the separated categories over to their associated next process locations. Doing this separation by hand is extremely tedious, time-consuming, and prone to error.

The hope is that automating part of this process would increase efficiency.

For an automation trial run, focused on filtering the Dermanson bean out of the other 6 kinds of mixed beans, we will see if a supervised machine-learning model can be used to correctly classify what is and is not a Dermason bean. Once separated, the Demason bean will be packaged up, and shipped out to retail stores across the county.

### Metric of Success:

In this case, the manufacturer has stated that it is most important to try to minimize the false positive rate of classification, shown by the Specificity score, or minimize the number of beans incorrectly categorized as a Dermason bean.

Here, the false negative rate, shown by the Recal score or beans incorrectly categorized as not a Dermason bean, is the more acceptable error metric because all beans not clearly identified as one of the 7 categories will go to the batch processing area where the mixture will be sold to an animal food manufacturer.

### Method of Extracting Data From Beans and Filtering Beans into Categories:

After the mixed wash, the beans will go down a conveyor belt that is equipped with a series of high-resolution cameras programmed to take pictures of all the individual beans from different angles. Once these pictures are taken, a computer will analyze the photos with a computer vision program to extract 12 different dimensional metrics such as the bean's area, perimeter, and roundness. This extracted multivariate data will then be put into a Supervised Machine Learning Algorithm in an attempt to classify each bean into its correct category.

### Utilized Machine Learning Technologies:

We will be utilizing Decision Trees and Logistic Regression models paired with Stratified-K-Fold cross-validation techniques and grid searches to optimize and tailor these machine-learning models to the manufacturer's requested metric of success. Specifically One Verus All classification will be used to maximize Specificity and Precision, while not letting Recal drop too low.

## The Data:

### Data Source and Data Use:

Source: "Dry Bean." UCI Machine Learning Repository, 2020, https://doi.org/10.24432/C50S4B.

To simulate this multivariate data extraction I used the "Dry Bean" dataset from the UCI Machine Learning Repository (cited above).

This Dataset contains over 13,000 data instances of beans that have had multivariate data extracted from pictures taken of them via a computer vision system. These data instances are made up of 7 different types of registered dry beans, with each instance having 16 features that describe different dimensional and shape-form metrics the bean exhibits.

Here the "Class" variable, which dictates which of the 7 types of beans each instance is, will be manipulated to create a new variable called "Dermason" that states if the data instance is or is not a Dermason bean. This new "Dermason" variable will be the target-dependent variable and will enable us to use One Versus All classification.

The other 16 variables, listed below, make up the independent variables we will use to train the machine-learning models to classify each data instance as either "Yes" a Dermason bean, or "No" not a Dermason bean. Each of these 16 variables aligns perfectly with potential dimensional metrics that could be extracted should the beans be going down a conveyor belt with cameras and a computer vision program programmed to extract dimensional information from the individual bean pictures they process.

Independent Variables:

1.) Area (A): The area of a bean zone and the number of pixels within its boundaries.
2.) Perimeter (P): Bean circumference is defined as the length of its border.
3.) Major axis length (L): The distance between the ends of the longest line that can be drawn from a bean.
4.) Minor axis length (l): The longest line that can be drawn from the bean while standing perpendicular to the main axis.
5.) Aspect ratio (K): Defines the relationship between L and l.
6.) Eccentricity (Ec): Eccentricity of the ellipse having the same moments as the region.
7.) Convex area (C): Number of pixels in the smallest convex polygon that can contain the area of a bean seed.
8.) Equivalent diameter (Ed): The diameter of a circle having the same area as a bean seed area.
9.) Extent (Ex): The ratio of the pixels in the bounding box to the bean area.
10.) Solidity (S): Also known as convexity. The ratio of the pixels in the convex shell to those found in beans.
11.) Roundness (R): Calculated with the following formula: (4piA)/(P^2)
12.) Compactness (CO): Measures the roundness of an object: Ed/L
13.) ShapeFactor1 (SF1)
14.) ShapeFactor2 (SF2)
15.) ShapeFactor3 (SF3)
16.) ShapeFactor4 (SF4)

### Data limitations

There are some limitations to this dataset that I would like to note:

Without seeing the actual pictures that this dataset used to extract each bean's dimensionality and shape factor we have no way of knowing for sure if each stated bean classification is actually true. Instead, we are only operating under the unverified assumption that these classifications are true.

The same can be said for calculations of the bean dimensionality and shape factor variables. Without being able to see the code behind how the computer vision and calculations work, we are only operating under the unverified assumption that these dimensional measurements and shape factors are true.

Many factors and attributes can be used to describe the properties of a dried bean. Off the bat, size, weight, and color are three common metrics that I can think of. This dataset only takes into account metrics that I would put under the size umbrella, leaving other potentially very useful information on the table.

Given that information, we can only state that this model will be classifying the type of bean based on unverified dimensional metrics that we assume to be true.

## Bring in the data and preview it

![Bean Datase](./pictures/Bean_dataset.png 'Bean Dataset')

## Feature Engineering

Check to see what proportion of the data each of the 7 types of beans takes up. From this take the most frequent bean type and manipulate the data so that it can be used for One Versus All classification of that highest proportion bean type.

Dermason is the most frequent bean type. Manipulate the data to make that the new target column.

## Split the data into train and test subsets for model evaluation and training

## Data Preprocessing

### Data Distribution Normalization

Check out the distribution of the existing training dataset. Its distribution, if skewed, may result in problematic generalization of under-represented classes (like Dermason Bean), model overfitting, or skewed performance metrics of the binary classifier model.

![existing data](./pictures/data_pre_normalization.png 'pre_normalization')

Nearly all of the columns in the training dataset are skewed. See if running the right skewed data through Log Transformations, and running the left skewed data through a Box-Cox Transformation can normalize their distributions.

![existing data](./pictures/data_post_normalization.png 'pre_normalization')

Although not perfect, this technique has greatly improved the distrobutions of the data.

### Data value scaling

Using a scaler to scale the values of the independent variables can help some models like Logistic Regression treat all values will equal importance. For example Area is currently counted in the 10,000's range, whereas Shape Factor 1 is counted in the .001's range. Due to Area's number being inherently larger, it may receive a higher weight when taking different factors into account. Scaling the data would help place all the variables on equal footing, potentially allowing the model to make more valuable predictions.

### Data SMOTEing and Random Undersampling

Using SMOTEing and Random Undersampling can both help and hurt datasets with unbalanced dependant target variables (like Dermason with ~25% of the datapoints and Not-Dermason with 75% of the datapoints). On the positive side doing these two things can create more synthetic instances of the minority class (Dermason) can cut back the inputted proportion of the majority class (Not-Dermason), bringing the proportion closer to 50%-50%. On the downside, doing both of these things could also create instances of minority class data combinations that are not natural, and potentially eliminate previous training data that was crucial to the model learning.

## Model instantiation, tuning, and evaluation 

Creation of a class that allows us to pass in different classification models, preprocess the data passed into the models, tune the hyperparameters of the models, and view how effective these models are at achieving the manufacturer's requested success metrics.

This class will use cross-validation to help us understand how the models will react to "unseen" data, and not just overfit on the training data during the tuning stage of the model selection process. In addition to this, since our target class (Dermason) is imbalanced in our dataset (only takes up roughly 25% of the instances), this class will specifically use Stratified K-fold Cross Validation to make sure that each fold the dataset is broken into maintains this similar proportion (~25%), making the folds more representative of the original dataset and easier to compare to each other.

Data preprocessing option selection will be available through choosing the dataset you pass in (normalized distribution or non), whether you want the data scaled/ what kind of scaler, and whether you would like the data to be SMOTE'd / Randomly Undersampled to help the training data's class imbalance move closer to 50%-50% from the existing 25%-75%.

Model hyperparameter tuning options will be available by passing any hyperparameter values the classification model may have into the keyword argument (model_kwargs).

As a representation of how these models are performing in regards to the manufacturer's requested metric of success, the passed in models will output the following:

1) The average Specificity score across the folds for the training and validation datasets.
2) The average Precision score across the folds for the training and validation datasets.
3) The average Recall score across the folds for the training and validation datasets.
4) A confusion matrix for the training and validation sets containing the average TN, FP, FN, TP rates

### Baseline classification models

#### Run a baseline Logistic Regression model to get the unpreprocessed and untuned performance results

![LR Basemodel](./pictures/LR_Basemodel.png 'LR basemodel')

Based off of the baseline model's performance results we can say that the training model does not seem to be overfitting on the data and generalizes well to unseen data. With a 96.187% validation specificity score we can say that the model is already very good at identifying what is not a Dermason bean. The validation precision score of 88.769% tells us that out off all the beans the model predicted to be a Dermason bean roughly 88% of them actually are. These are not a bad scores, although a second filter may be needed on the production line to eliminate the small percentage of non-Dermason beans that were predicted to be Dermason beans. The recal is the lowest of the three metrics, but is still fairly high at 85.613%, and due to this being the manufacturer's least important metric of the three, correctly identifying 85% of the actual Dermason beans is not that bad given all the Dermason beans we failed to identify can still be sold as animal food.

#### Run a baseline Decision Tree model to get the unpreprocessed and untuned performance results

![DT Basemodel](./pictures/DT_Basemodel.png 'DT basemodel')

Based off of the Decision Tree (DT) baseline model's performance results we can say that the training model is overfitting on the training data and does not perform as well with unseen data as it does with the training dats. However, with a 96.352% validation specificity score we can say that this model performs better than the baseline Logistic Regression (LR) model at identifying what is not a Dermason bean. The validation precision score of 89.575% tells us that out off all the beans the model predicted to be a Dermason bean roughly 89% of them actually are, which is higher than the LR's score as well. These are not a bad scores, although a second filter may still be needed on the production line to eliminate the small percentage of non-Dermason beans that were predicted to be Dermason beans. The recal score is the still lowest of the three metrics, but is still fairly high at 89.389% and is roughly 4% higher than the LR's score. Due to recal still being the manufacturer's least important metric of the three, correctly identifying 89% of the actual Dermason beans is not that bad given all the Dermason beans we failed to identify can still be sold as animal food.

### Tuned classification models

Now that we have a general idea how well each model will perform when presented with unseen data, let's run grid searches on the models to find out which preprocessing and hyperparameter tuning will result in the most effective versions of these models in reference to the manufacturers requested success metrics.

## Run a grid search for preprocessing and hyperparameter tuning the Logistic Regression model

Create a new dataframe going over the performance results of each of these iterations and the models' inputs.

Sort the new dataframe to show which models had the highest specificity rating.

![LR Gridsearch](./pictures/LR_Grid_Search.png 'LR Gridsearch')

Even though there are some instances where the specificity score is technically perfect we are not interested in those instances due to their major loss of the recal score (correctly identifying less than 1% of the Dermason beans as Dermason beans). Due to this we will scroll down the list to find a more balances instance where there is not so much loss on recal.

Optimizing for Specificity and Precision, without sacrificing too much recall, we have located an instance where the following average validation set scores are achieved:

1) Specificity: 97.9831%
2) Precision: 92.7953%
3) Recal: 81.720%

This was achieved through the following:

Data Preprocessing:

1) Scaler: None
2) Data distribution normalization through Box-Cox and Log transformation: True
3) Class imbalance redistribution through SMOTE and Random Undersampling: False

Model Hyperparameter Tuning:

1) C= 10
2) solver= liblinear
3) fit_intercept= True
4) penalty= l2

Now, sort the new dataframe to show which models had the highest precision rating to see if this gives us a better answer.

This seemingly just gets us to our above-desired instance faster, displaying the same called-out optimized instance that we just called out above. We will use this call to look for the optimized Decision Tree results instead of sorting by specificity.

### Display the results of the optimized Logistic Regression model and compare to previous best model

![LR Tuned](./pictures/LR_Tuned.png 'LR Tuned')

We can see that this new tuned Logistic regression model is performing better than the previous most effective model (Baseline Decision Tree). This new LR model's validation specificity improved by almost 2%, and its precision improved by almost 4% over the previous best model. The only downside is that recal took about an 8% hit, but given the manufacturer's context for efficiency (maximize specificity and precision), this is an acceptable loss.

This is now considered the most effective model given the context.

Now let's run an optimization grid search on the Decision tree to see if it can out perform the tunes Logistic Regression model.

## Run a grid search for preprocessing and hyperparameter tuning the Decision Tree model

Create a new dataframe going over the performance results of each of these iterations and the models' inputs.

Sort the new dataframe to show which models had the highest precision rating.

![DT Gridsearch](./pictures/DT_grid_search.png 'DT Gridsearch')

Sorting by precision immediately takes us to a well-balanced specificty/precision/recal instance.

Optimizing for Specificity and Precision, without sacrificing too much recall, we have located an instance where the following average validation set scores are achieved:

1) Specificity: 98.6774%
2) Precision: 95.5633%
3) Recal: 80.6682%

This was achieved through the following:

Data Preprocessing:

1) Scaler: None
2) Data distribution normalization through Box-Cox and Log transformation: False
3) Class imbalance redistribution through SMOTE and Random Undersampling: True

Model Hyperparameter Tuning:

1) max_depth= 2
2) min_samples_split= 0.05
3) criterion= gini
4) min_samples_leaf= 0.1425
5) max_features= 4

### Display the results of the optimized Decision Tree model and compare to previous best model

![DT Tuned](./pictures/DT_tuned_model.png 'DT Tuned')

We can see that this new tuned Decision Tree model is performing better than the previous most effective model (Tuned Logistic Regression). Not only did this tuned model get rid of most of the baseline Decision Tree's overfitting, but its validation specificity improved by almost 1% over the previous best model, and its precision improved over 2% over the previous best model. The only downside is that recal took about an 0.05% hit compared to the previous best model, but given the manufacturer's context for efficiency (maximize specificity and precision), this is an acceptable loss.

This is now considered the most effective model given the context.

Now lets run this new best model on the X_test data hold out we saved from the original train test split, and evaluate the results

## Run this final tuned Decision Tree Model on the X_test data holdout and evaluate the results

I created a new model evaluating class by modifying the above ModelWithCv class so that it runs any inputted model on the final X_test data, instead of using Stratified K-fold Cross Validation on the training data, while keeping most other class features and functionality constant. This class will also be able to produce an accuracy score, which is not 100% relevant to the manufacturer's requested success metrics, but could still be interesting to know. I also removed the scaler input because the model we will be plugging into this will not be using scaling data preprocessing.

Input the final model details into the new X_test model evaluator:

model_instantiator= DecisionTreeClassifier

Data Preprocessing:

1) Scaler: None
2) Data distribution normalization through Box-Cox and Log transformation: False
3) Class imbalance redistribution through SMOTE and Random Undersampling: True

Model Hyperparameter Tuning:

1) max_depth= 2
2) min_samples_split= 0.05
3) criterion= gini
4) min_samples_leaf= 0.1425
5) max_features= 4

![Final Model](./pictures/Final_Model.png 'Final Model')

## Conclusions

My conclusion from the meta-analysis is that this food manufacturer could use binary classification via supervised machine learning in their production line to start to implement process automation that would cut down on manual labor time and potentially improve their manufacturing efficiency.

In this specific test case area of concern, the Tuned Decision Tree Binary classification model can be used to separate Dermason beans from the mixture of other 6 other beans as the batch heads down the conveyor belt. If the classifier determines the bean is indeed a Dermason bean, an actuated pusher could be utilized to push the bean off of the belt and over to the Dermason processing area.

1) Since Speceficity and Precision were the requested optimized metrics for success, and recal being the metric with the highest allowable error rate, we can say the following:

2) Given this model's Specificity rating of 98.286% on the testing data we can say that this model is extremely good at determining what is not a Dermason bean. This means that less than 2% of the time, a non-Dermason bean is incorrectly classified as a Dermason bean.

3) Given this model's precision rating of 94.444% on the testing data, we can say that when the model predicts a bean to be a Dermason bean, there is a high likelihood that it actually is a Dermason bean. This means that out of all the beans the model predicts as Dermason, less than 6% are actually non-Dermason beans.

Given this model's Recal rating of 78.108% on the testing data we can say that the model is decent at being able to identify all of the dermason beans going down the conveyor belt. This means that out of all the beans going down the conveyor that are actually Dermason beans, the model will correctly classify slightly over 78% of those beans as Dermason beans. Even though this score is not bad, it is still lower than the model's extremely high Specificity and Precision scores, but in this context that is okay because we would rather minimize the errors for beans that are classified as Dermason beans, and all beans not distinctly identified as a specific classification will be sent to the mixed batch process location where they can still be sold for a profit.

To deal with the small percentage of beans that are incorrectly classified as Dermason beans after going through the Decision Tree classification process, the manufacturer could put these beans through a second manual double-checking filter to remove the small number of incorrect classifications, thus significantly cutting down the time and effort put into manual sorting on the production line.

## Next Steps

Here are three potential next steps that the manufacturer can take to further improve their system via automation:

1) Add additional types of sensors to the bean conveyor belt. Currently, this model is only run using dimensional multivariate data from computer vision processing. Beans have many more easily quantifiable attributes like weight and color that could be taken into account when trying to classify the type of bean. Equipping the conveyor with sensors that can extract this currently unquantified data could improve the effectiveness of this model.

2) Instead of running a bunch of individual one vs all classifications to sort the different types of beans, we could make a multiclassification model that classifies all the different kinds of beans at once.

3) Given more metrics and context other than product dimensions, a classification model could be created to separate the products by grade and quality. This could enable the manufacturer to separate one product into different tears such as: Medicinal Grade, Human Consumption, Animal Consumption, and Throw Away.






# Repository Structure

├── my_data

├── pictures

├── .gitignore

├── Automation_Integration_Via_Dermason_Bean_Classification.ipynb

├── Automation_Integration_Via_Dermason_Bean_Classification.pdf

├── Automation_Machine_Learning_Slides.pdf

└── README.md





