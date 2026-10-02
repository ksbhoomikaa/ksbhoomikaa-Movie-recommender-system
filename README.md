This group project is a desktop-based Hybrid Movie Recommendation System that uses machine learning to suggest personalized content. By combining Content-Based (Summary and genre) and Rating-Based Filtering, we get a balanced list of recommendations that account for both the intrinsic features of a movie and its popularity among users.

Problem statement:
Modern digital libraries face information overload and the "cold start" problem, where traditional collaborative filtering fails to recommend new movies or serve new users due to a lack of historical data. Additionally, content-based systems often create "echo chambers" by limiting variety, while a lack of visual interfaces makes current recommendation outputs difficult for average users to interpret.

We tried to combine 2 recommender systems. 
The first one was recommending based on the actual content and genre of the movie, using TF-IDF to convert movie overviews and genres into numerical vectors, and calculating Cosine Similarity between these vectors. This was Content-Based Filtering.
The second was Collaborative Filtering, which uses the K-Nearest Neighbors (KNN) algorithm on a user-rating matrix to find movies that the same group of people liked, regardless of the plot.
Both gave their own score, which we combined to give the final rating.
We decided to give more weightage to the content based recommeder, and the final score depended 60% on content-based and 40% on collaborative.
We also learnt how to clean the data by removing years at the end of movie names and learning how to combine both recommenders and how to utilise all datasets.
We also used WxPython to build the GUI for the project.
We explored what an api key is and used it to access the IMDb website to get posters for our movies.
