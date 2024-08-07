# Miniset-DenseSENet: An Effective Model Framework for ALS Diagnosis

**For a detailed introduction, please refer to [Poster.pdf](./Poster.pdf).**

In this study, we exploit the combination of transfer learning and attention mechanism to develop an effective model framework named Miniset-DenseSENet. The model leverages the powerful feature extraction capabilities of pre-trained DenseNet121 and combines SE modules to enhance feature representation. By focusing on the most relevant features in the dataset, our approach aims to improve the diagnosis and understanding of ALS. Our main contributions are as follows:

1. **Postmortem Brain Image Analysis**:
   - We explored the differences in postmortem brain images between individuals with and without ALS, as well as ALS patients with and without cognitive impairment.

2. **Novel Model Framework**:
   - We developed a novel deep learning model framework named Miniset-DenseSENet, which integrates DenseNet121 and SE modules. This framework leverages transfer learning to enhance feature extraction and incorporates attention mechanisms to improve the quality of feature representations.

3. **Comprehensive Evaluation**:
   - We performed a comprehensive evaluation of various transfer learning models and their attention mechanism variants on the ALS classification problem. Our experiments included a wide range of models such as baseline CNN, ResNet18, DenseNet121, ResNet18+SE, ResNet18+CBAM, and Miniset-DenseSENet, providing a detailed performance comparison and identifying the most effective approaches.

4. **Visual Explanations with Grad-CAM**:
   - We provided visual explanations for the model’s classification decisions using Gradient-weighted Class Activation Mapping (Grad-CAM). These visualizations help to interpret the model’s behavior and highlight the critical regions in brain images that influence the classification results, thereby enhancing the transparency and trustworthiness of the model.

## Abstract

Amyotrophic lateral sclerosis (ALS) is a progressive neurological disease leading to motor function deterioration and eventually respiratory failure and death. Identifying early diagnostic biomarkers for sporadic ALS is challenging due to the undefined risk population, making large datasets difficult to obtain. In this project, we used a dataset of 190 autopsy brain images from the Gregory Laboratory at the University of Aberdeen to explore convolutional neural networks for ALS classification. We developed Miniset-DenseSENet, a model combining DenseNet121 with an Squeeze-and-Excitation (SE) attention mechanism, to analyze these images. Our model aims to distinguish ALS patients from control groups and identify cognitive impairments in ALS patients. Miniset-DenseSENet outperformed other transfer learning models and attention mechanisms, achieving 97.37% accuracy, an average MCC of 0.84, and sensitivity and specificity of 1 and 0.95, respectively. These results highlight the potential of integrating transfer learning and attention mechanisms to improve diagnostic accuracy and provide new insights into cognitive impairment associated with ALS. This approach not only improves the ability to identify ALS features, but also provides a new perspective for distinguishing ALS patients with and without cognitive impairment.

## Keywords

Amyotrophic lateral sclerosis, TDP-43 protein, Cognitive impairment, Transfer learning, Attention mechanisms.
