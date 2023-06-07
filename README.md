# Dog Diseases Classification Modeling
This repository contains 2 folders to store our notebook for 2 different models. Both models trained using TensorFlow, and the dataset can be found in each model's folder.

# Dog Symptoms Model
The model aim to accurately classify 12 dog diseases based on symptoms given using ANN and it has been implemented in our application. The tabular dataset was forked from [this repository](https://github.com/1zuu/Doggy-Disease-Detection/blob/master/data/symtomdata.csv)
| **Information** | **Value** |
| --- | --- |
| Preprocessing Technique | One Hot Encoding |
| Model Structure | Combination of Dense layers and Dropout layers |
| Model Input | List of symptomps |
| Model Output | Classification of Dog Diseases |

# Dog Skin Disease Model 
The model aim to classify 4 dog skin diseases using CNN. The image dataset were collected independently by manually downloading it from Google Images. We didn't conduct image scraping due to data scarcity.
Our model performed wonderfully during training, yet it fails to produce useful results during testing. Even tough we have serious overvitting issue, we kept implementing it in our application with some considerations. 
| **Information** | **Value** |
| --- | --- |
| Preprocessing Technique | Offline augmentation, Image scaling |
| Model Structure | Classic CNN |
| Model Input | Image |
| Model Output | Classification of Dog Skin Diseases |
