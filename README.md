# Predicting Molecular properties

This assignment aims to predict the magnetic interaction i.e. 'Scalar coupling constant' between the atoms constituting a molecule. Whereas, the position of atoms in 3D space is given. These positions are represented as x, y, z Cartesian co-ordinates.

By using the state-of-the-art methods from quantum mechanics, it is possible to accurately calculate scalar coupling constants given only a 3D molecular structure as input. However, these quantum mechanics calculations are extremely expensive (days or weeks per molecule), and therefore have limited applicability in day-to-day workflows. A fast and reliable method to predict these interactions will allow medicinal chemists to gain structural insights faster and cheaper, enabling scientists to understand how the 3D chemical structure of a molecule affects its properties and behaviour.

The determining factor of these magnetic interactions are respective distances between the atoms. Given the 3D positions of atoms in space i.e. Cartesian co-ordinates x, y, z one can measure the distance by Euclidean formula. These distances are then fed as feature vectors against the scalar coupling co-efficient to train and support vector regression machine.

The program is developed in Python and following are the dependencies:

**Dependencies**

 - Scikit-Learn - For Machine Learning
 - Pandas - For csv file reading, processing etc.
 -  NumPy - For array manipulation
 
 **Questions**
 
 **1.  Can  we  use normalization for some features?**
     Feature Normalization is considered to be the data pre-processing step. The purpose of this step to make the data set in a certain range. This step is     essential most of the time, because of the nature of real-world dataset. However, this step should be avoided if the data is in a specified range/scale.One can experiment and chose the type of normalization that produces the best results. The normalization used in this program is used from the build-in function *scklearn.preprocessing.StandardScalar*, that measures the euclidean distance and normalize the features that are dominating and out of bound.
    
**2.  Why you selected the given model and what other models you can try?**
In this assignment, Support Vector Regression and Linear Regression is chosen. As the nature of the labels was continuous so a regression model must be used instead of classifier. Both of these models are used from *sklearn library*. However, an extensive experimentation can be carried out and different kernels, can be chosen to check the effect on output or accuracy. Similarly, the a log grid search can be performed on the values of cost and gamma to get the best possible values. For this project a fixed value of cost and gamma is used. 
    
**3.  Which evaluation metric is suitable for the problem and why?**
In my opinion, simple accuracy formula is enough evaluation metric for this kind of problem.

**4.  Why the model (if the model isn't working well) not producing good results?**
There is a possible arrangement/alignment error between the data and the labels. It is producing reasonably good results for the first 10 entries . That is the accuracy is almost equal to 100%. But as we are increasing the number of data points, its accuracy is decreasing gradually. Another possible reason for decreasing accuracy is data normalization, i.e. different technique must be used to data scale. Also, there is some additional data given that can be added as features so that the accuracy can be improved. 


 
