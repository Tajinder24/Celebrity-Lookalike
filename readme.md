# Find your look-alike celebrity face 

The dataset consist of 100 images each of 80 bollywood actors and actresses. In the first step, all the images are converted to a pickle file. In the next step, all the images are read from the pickle file and passed to the VGGFace module with resolution size 224x224. This module extracts the important features from the image and store it in pickle format. Now when the user gives an input image, the important features are extracted from this image using MTCNN detector and the feature is macthed with the training data image features using cosine similarity. The image that has the highest similarity is returned as output. Streamlit is used for local server deployment.

## Important modules

### create_image_pickle_file.py
It converts the training data images to pickle file format and stores it in "*artifacts/pickle_format_data/image_pickle_file.pkl*" file

### feature_extractor.py
It extractes the features from the training data images and stores it in "*artifacts/extracted_features/embedding.pkl*" file 

### app.py
This file consist of the code for local server deployment and prediction for the input image.

## Important commands
```
pip install -r requirements.txt
```
For package installation 

```
python create_image_pickle_file.py
```
To create pickle file for images. Run this command only once and it will store the data in the specified folder

```
python feature_extractor.py
```
To extract features. Run this command only once and it will store the data in the specified folder.

```
streamlit run app.py
```
To run the local server. This command starts the server and you can then upload image on your browser. The result will be displayed there.