Problem Analysis
The dataset is characterized by 13759 RGB images
with shape 96x96, representing cells. It is a MultiClassification problem with 8 classes: Basophil,
Eosinophil, Erythroblast, Immature granulocytes,
Lymphocyte, Monocyte, Neutrophil, Platelet.
An initial analysis of the dataset reveals an imbalance in the distribution of the samples across the
classes, possible cause of a biased model performance. The class-specific statistical analysis of the
images reveals the presence of outliers for each label. In addition, the Monocyte class has a further
deviation from the expected distribution.
Finally, the attention to details in biological applications is an important aspect to consider as certain
blood cell types may exhibit morphological similarities. The low resolution of the dataset images might
affect the classification problem 


3.1. Data Preparation
The first step involved identifying and removing outliers. For each class, the mean and standard deviation of pixel values are computed and
instances deviating more than 1.5 times the standard deviation from the class mean are flagged as
outliers. The next step involved augmentation.
The dataset has been tripled preserving the class
proportions. To do so, it has been used a combination of Keras ImageDataGenerator, that involves
standard transformations such as rotation, flipping,
shifting, zooming, and Keras RandAugment [1],
that applies 3 additional random transformations
on each new image.
The dataset was then split into training, validation
and test subsets maintaining the same class distribution. Also, the training set was balanced through
the use of augmentation and oversampling in order
to have the same number of instances of each class
and to ensure that the model does not overfit to
majority classes.

3.2: Model design
In order to increase the generalization capability and the robustness, the proposed model is an
EfficientNetV2-B0 pre-trained on ImageNet dataset, which was further fine-tuned through Transfer Learning. An augmentation layer was
added at the beginning of the architecture to introduce variability in the input data during training.
The output block was customized to include Global Average Pooling, Group Normalization, Dropout,
and a Dense layer to adapt it to the specific task.
Fine-tuning was performed with the initial 50 layers
of the base model frozen to preserve the pre-trained
features, while allowing the upper layers to adjust
and specialize for the new classification problem.
The model was compiled using Adam optimizer
and categorical cross-entropy loss
