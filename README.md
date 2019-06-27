# Predicting Molecular properties

This assignment aims to predict the magnetic interaction i.e. 'Scalar coupling constant' between the atoms constituting a molecule. Whereas, the position of atoms in 3D space is given. These positions are represented as x, y, z Cartesian co-ordinates.

By using the state-of-the-art methods from quantum mechanics, it is possible to accurately calculate scalar coupling constants given only a 3D molecular structure as input. However, these quantum mechanics calculations are extremely expensive (days or weeks per molecule), and therefore have limited applicability in day-to-day workflows. A fast and reliable method to predict these interactions will allow medicinal chemists to gain structural insights faster and cheaper, enabling scientists to understand how the 3D chemical structure of a molecule affects its properties and behaviour.

The determining factor of these magnetic interactions are respective distances between the atoms. Given the 3D positions of atoms in space i.e. Cartesian co-ordinates x, y, z one can measure the distance by Euclidean formula. These distances are then fed as feature vectors against the scalar coupling co-efficient to train and support vector regression machine.

The program is developed in Python and following are the dependencies:

**Dependencies**

 - Scikit-Learn - For Machine Learning
 - Pandas - For csv file reading, processing etc.
 -  NumPy - For array manipulation
