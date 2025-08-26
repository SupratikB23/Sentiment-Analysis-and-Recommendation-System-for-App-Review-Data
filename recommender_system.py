import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_recommender(app_reviews_df):
    grouped_reviews = app_reviews_df.groupby('app_id')['review_text']
    combined_reviews = grouped_reviews.apply(lambda texts: ' '.join(texts))
    app_texts_df = combined_reviews.reset_index()

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(app_texts_df['review_text'])

    cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    app_id_to_index = pd.Series(data=app_texts_df.index.values, index=app_texts_df['app_id'])
    index_to_app_id = pd.Series(data=app_texts_df['app_id'].values, index=app_texts_df.index.values)

    average_scores = app_reviews_df.groupby('app_id')['review_score']
    avg_scores = average_scores.mean()

    return cosine_sim_matrix, app_id_to_index, index_to_app_id, avg_scores


def recommend_apps(app_id, cosine_sim_matrix, app_id_to_index, index_to_app_id, avg_scores, top_n=20):
    if app_id not in app_id_to_index:
        return pd.DataFrame()

    idx = app_id_to_index[app_id]
    sim_scores = cosine_sim_matrix[idx]

    sorted_indices = sim_scores.argsort()[::-1]

    top_indices = []
    for i in sorted_indices:
        if i != idx:
            top_indices.append(i)
        if len(top_indices) == top_n:
            break

    recommended_app_ids = [index_to_app_id[i] for i in top_indices]
    similarity_scores = [sim_scores[i] for i in top_indices]
    avg_review_scores = [avg_scores[app] if app in avg_scores else 0.0 for app in recommended_app_ids]

    result_df = pd.DataFrame({
        'app_id': recommended_app_ids,
        'similarity_score': similarity_scores,
        'avg_review_score': avg_review_scores})

    result_df = result_df.sort_values(by=['similarity_score', 'avg_review_score'], ascending=False)
    return result_df

if __name__ == "__main__":
    app_reviews_df = pd.read_csv("apps_reviews.csv", index_col=0)

    cosine_sim_matrix, app_id_to_index, index_to_app_id, avg_scores = build_recommender(app_reviews_df)

    recommendations = recommend_apps(1, cosine_sim_matrix, app_id_to_index, index_to_app_id, avg_scores, top_n=10)
    print(recommendations)
