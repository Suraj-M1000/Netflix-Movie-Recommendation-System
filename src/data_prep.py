import numpy
import pandas

def clean_netflix_data(netflix_dataset, movie_quantile = 0.6, user_quantile = 0.6):
    netflix_dataset = netflix_dataset.copy()
    is_marker = netflix_dataset['Cust_Id'].str.contains(":")
    netflix_dataset['Movie_id'] = numpy.where(is_marker, netflix_dataset['Cust_Id'], numpy.nan)
    netflix_dataset['Movie_id'] = netflix_dataset['Movie_id'].ffill()
    netflix_dataset['Movie_id'] = netflix_dataset['Movie_id'].str.replace(":","") 
    netflix_dataset['Movie_id'] = netflix_dataset['Movie_id'].astype('int')
    netflix_dataset = netflix_dataset[netflix_dataset['Rating'].notna()]
    netflix_dataset.rename(columns={'Cust_Id': 'User_Id'}, inplace=True)
    netflix_dataset.reset_index(drop=True, inplace=True)
    netflix_dataset['User_Id'] = netflix_dataset['User_Id'].astype("int")

    # Prefiltering - What we had did in the Notebook..
    movies_rate_summary = netflix_dataset.groupby('Movie_id')["Rating"].count()
    movies_rate_summary = pandas.DataFrame(movies_rate_summary)
    movie_benchmark = round(movies_rate_summary['Rating'].quantile(movie_quantile), 0)
    non_drop_movie_list  = movies_rate_summary[movies_rate_summary['Rating'] >= movie_benchmark].index


    dataset_cust_summary=netflix_dataset.groupby('User_Id')['Rating'].agg(["count"])
    cust_benchmark=round(dataset_cust_summary['count'].quantile(user_quantile),0)
    non_drop_cust_list=dataset_cust_summary[dataset_cust_summary['count'] >= cust_benchmark].index

    netflix_dataset = netflix_dataset[netflix_dataset['Movie_id'].isin(non_drop_movie_list)]
    netflix_dataset = netflix_dataset[netflix_dataset['User_Id'].isin(non_drop_cust_list)]

    netflix_dataset.reset_index(drop=True, inplace=True)
    return netflix_dataset, non_drop_movie_list, non_drop_cust_list

