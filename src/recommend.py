

def recommend_for_user(user_id, model, movie_rating_df, movie_title_df, keep_movie_ids, n=10):
    if user_id not in movie_rating_df['User_Id'].values:
        print(f"User {user_id} not found in dataset.")
        return None

    user_ratings = movie_rating_df[movie_rating_df['User_Id'] == user_id]
    user_rated_movies = user_ratings['Movie_id']

    candidates = movie_title_df[movie_title_df['Movie_Id'].isin(keep_movie_ids)].copy()
    candidates = candidates[~candidates['Movie_Id'].isin(user_rated_movies)]
    candidates['User_Preferences_score'] = candidates['Movie_Id'].apply(
        lambda x: model.predict(user_id, x).est
    )
    return candidates.sort_values(by='User_Preferences_score', ascending=False).head(n)

