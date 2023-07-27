Brain Tumor Classification using Deep Learning

This project aims to build a robust Deep Neural Network (DNN) model to classify brain tumor types from Magnetic Resonance Imaging (MRI) scans. The dataset used consists of four tumor types: Glioma tumor, No Tumor, Pituitary tumor, and Meningioma tumor. The model leverages Convolutional Neural Networks (CNNs) and Dense Neural Networks to effectively learn intricate features from the MRI images. The primary goal is to achieve high accuracy in tumor classification, aiding in early and accurate diagnosis.

Data Description:

THE DATA IS NOT COLLECTED BY ME , AND THE DATA was downloaded from here - https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri?select=Testing

The dataset contains MRI images of the brain, collected from various sources, and preprocessed to ensure uniformity and standardization. Each image is labeled with one of the four tumor types, allowing supervised training of the DNN model. The dataset is split into training and testing subsets to evaluate the model's generalization performance.

Model Architecture:
The model architecture comprises multiple convolutional and dense layers. The CNN layers act as feature extractors, capturing local and spatial patterns in the MRI images. Subsequent dense layers perform high-level feature learning and classification. Batch normalization and dropout are utilized to prevent overfitting and promote better generalization. Additionally, learning rate scheduling is employed to adaptively update the learning rate during training, enhancing convergence and optimizing performance.

Training and Results:
The model is trained using the training subset with labeled tumor images. During training, data augmentation techniques like shear, zoom, and horizontal flip are applied to augment the dataset, further enhancing model robustness. The model is optimized using the Adam optimizer and trained to minimize binary cross-entropy loss. After training, the model is evaluated on the testing subset to assess its performance on unseen data.

Accuracy and Performance:
The trained model demonstrates promising results, achieving an accuracy of 88.05% on the testing data. The achieved accuracy suggests the model's ability to accurately classify brain tumor types based on MRI scans. The model's high accuracy and robustness indicate its potential to aid neurobiologists and radiologists in accurate and timely tumor diagnosis, enabling appropriate medical interventions.

In conclusion, this Deep Learning-based Brain Tumor Classification model exhibits impressive performance in detecting different brain tumor types from MRI scans. The combination of CNNs, DNNs, data augmentation, and learning rate scheduling techniques contributes to the model's success. The accurate classification of brain tumor types can significantly impact early diagnosis, personalized treatment, and improved patient outcomes, making this model a valuable tool in the field of neurobiology and medical imaging analysis.



Summary:
This project presents a state-of-the-art Deep Neural Network (DNN) model for brain tumor classification based on Magnetic Resonance Imaging (MRI) scans. The model utilizes Convolutional Neural Networks (CNNs) for feature extraction, Dense Neural Networks for high-level feature learning, and employs data augmentation techniques to enhance dataset diversity.

The brain tumor classification model utilized deep neural networks (DNN), convolutional neural networks (CNN), data augmentation, learning rate scheduling, categorical cross-entropy loss, Adam optimizer, image preprocessing, one-hot encoding, and model saving/loading techniques to achieve accurate multi-class tumor classification from MRI scans.

The classification is between No Tumor, Pituitary tumor, Glioma and Meningioma on a scale of 0 to 1

 Additionally, learning rate scheduling optimizes training convergence. The model showcases an impressive 88.05% accuracy, offering potential clinical value in neurobiology and medical imaging analysis. Accurate tumor type classification can aid neurobiologists and radiologists in timely diagnosis and personalized treatment, positively impacting patient outcomes and advancing the field of computational neurobiology.

The computational tools used in building the brain tumor classification model include Python, Jupyter Notebook (IPython), TensorFlow, Keras, NumPy, Matplotlib, OpenCV, and scikit-learn. These tools were instrumental in implementing various machine learning and deep learning techniques, handling image data, and visualizing results for accurate tumor classification from MRI scans.


