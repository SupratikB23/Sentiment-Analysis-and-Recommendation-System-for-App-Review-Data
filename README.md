# Sentiment-Analysis-for-App-Review-Data-Recommendation-System

## Sentiment Classification: </br>
To build a sentiment classifier that can accurately analyze and predict the sentiment (positive, negative, or neutral) of user reviews collected from 1 million Google Play Store entries. This will help in understanding user satisfaction, detecting dissatisfaction trends, and enhancing app quality through actionable feedback.

## Personalized App Recommendation System:</br>
To develop a personalized app recommendation engine that combines both:</br>
a) Content-Based Filtering (based on the textual content of user reviews using natural language processing techniques)</br>
b) Collaborative Filtering (based on average user ratings and usage patterns across users and apps). </br>
The system will suggest relevant apps to users based on their preferences, behavior, and similar user interactions, ultimately improving app discovery and user engagement on the platform.

DATASET - [Play Market 2025 - 1M Reviews, 500+ Titles](https://www.kaggle.com/datasets/dmytrobuhai/play-market-2025-1m-reviews-500-titles)

## Model Architecture for Sentiment Analysis: </br>
A hybrid sequence model combining both Bidirectional LSTM and Bidirectional GRU layers are used. This architecture is particularly effective for extracting sequential dependencies and capturing the context of words in reviews.</br>
1. The model starts with an Embedding layer that converts word indices into dense vector representations of dimension 100.</br>
2. A Bidirectional LSTM layer with 128 units is used to process the sequences in both forward and backward directions, capturing long-range dependencies in text.</br>
3. The output of the LSTM is passed to a Bidirectional GRU layer with 64 units, which further captures sequential patterns and improves efficiency over stacked LSTMs.</br>
4. A Dropout layer with a rate of 0.5 is applied after the GRU to reduce overfitting by randomly deactivating 50% of the neurons during training.</br>
5. A Dense layer with 64 neurons and ReLU activation introduces non-linearity and helps in learning complex representations.</br>
6. The final output layer is a Dense layer with 3 neurons and softmax activation to classify the input into one of three sentiment categories: negative, neutral, or positive.</br>
7. The model is compiled using the Adam optimizer, with sparse categorical crossentropy as the loss function, and accuracy as the evaluation metric.</br>

## Recommendation pipeline: </br>
• Text Aggregation (Content Preparation): All the review texts are grouped per app ID. For each app, all reviews are combined into a single text string. This results in a single textual representation per app, which can be used for similarity comparison.</br>
• Text Vectorization (TF-IDF): This transformation captures the importance of each word in each app's review corpus while ignoring common stop words.</br>
• Similarity Computation: Using the TF-IDF matrix, pairwise cosine similarity is calculated between all apps. Cosine similarity helps determine how similar two apps are based on the angle between their vector representations. </br>
• Review Score Aggregation: For each app, the average review score is computed using the raw review data. This value is used to enhance the quality of the recommendations.</br>
• Recommendation Function: Given an app ID -> Its similarity scores to all other apps are retrieved -> Top N similar apps are selected (excluding the app itself) -> For each recommended app: The similarity score and average review score are collected -> A DataFrame is returned, sorted first by similarity score and then by average review score.</br>

## Evaulation & Results: </br>
<b>Sentiment Analysis:</b></br>
For evaluating the sentiment analysis model, accuracy was used as the primary metric. Accuracy measures the proportion of correctly predicted sentiment labels out of all predictions, providing a straightforward indicator of overall model performance. After training and testing the model, it achieved an accuracy of 0.711, meaning it correctly classified about <b> 71% </b> of the test reviews. This indicates a reasonably good performance in predicting whether a review’s sentiment is negative, neutral, or positive.</br>
</br>
<b>Recommendation System:</b> For evaluating the recommendation system, several metrics can be used to measure how well the system suggests relevant apps to users. In this project, the recommendation system allows the user to input a specific app ID for which they want recommendations, as well as specify the number of recommendations desired. The core evaluation metrics typically considered for recommendation systems include: </br>
a. Precision and Recall </br>
b. Similarity Scores </br>
c. Average Review Scores</br>

## Some Vizualizations: </br>
![image](https://github.com/user-attachments/assets/2f35153a-593c-4855-83d4-8264f49df62e) </br>
<b> Sentiment Analysis - HEATMAP </b> </br>

![image](https://github.com/user-attachments/assets/9b15829c-fee1-443f-9756-0a8290c36255) </br>
<b> Sentiment Analysis - SENTIMENT DISTRIBUTION </b> </br>

![image](https://github.com/user-attachments/assets/253fe73b-13f7-46ea-b5bf-34cf2463634f) </br>
<b> Recommendation System DEMO </b>

