from flask import Flask, request, render_template
from flask_cors import cross_origin
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from tmdbv3api import TMDb, Movie
import requests

app = Flask(__name__)

df = pd.read_csv('preprocessed_data.csv')
df_cache = pd.read_csv('cache_data.csv')
movie_list = list(df['movie_title'])

tmdb = TMDb()
tmdb.api_key = '408b5f69b58f03be6d4e81fcdb53074a'


def get_poster_link(title_list):
    tmdb_movie = Movie()

    dic_data = {"Movie_Title": [], "Poster_Links": [], "Tag_Line": []}

    for title in title_list:

        r_df = df_cache[df_cache['Title'] == title]
        try:
            if len(r_df) >= 1:
                dic_data["Movie_Title"].append(r_df['Movie_Title'].values[0])
                dic_data["Poster_Links"].append(r_df['Poster_Links'].values[0])
                dic_data["Tag_Line"].append(r_df['Tag_Line'].values[0])

            else:
                result = tmdb_movie.search(title)
                movie_id = result[0].id
                response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id, tmdb.api_key))
                data_json = response.json()

                movie_title = data_json['title']
                movie_poster_link = "https://image.tmdb.org/t/p/original" + data_json['poster_path']
                movie_tag_line = data_json['tagline']

                dic_data['Movie_Title'].append(movie_title)
                dic_data['Poster_Links'].append(movie_poster_link)
                dic_data['Tag_Line'].append(movie_tag_line)
        except:
            pass

    return dic_data


@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route('/', methods=['POST', 'GET'])
@cross_origin()
def recommendation():
    if request.method == 'POST':
        try:
            title = request.form['search']
            title = title.lower()
            cv = CountVectorizer()
            count_matrix = cv.fit_transform(df['comb'])

            cosine_sim = cosine_similarity(count_matrix)

            correct_title = get_close_matches(title, movie_list, n=3, cutoff=0.6)[0]

            idx = df['movie_title'][df['movie_title'] == correct_title].index[0]

            sim_score = list(enumerate(cosine_sim[idx]))

            sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)[0:15]

            suggested_movie_list = []
            for i in sim_score:
                movie_index = i[0]
                suggested_movie_list.append(df['movie_title'][movie_index])

            poster_title_link = get_poster_link(suggested_movie_list)
            return render_template('recommended.html', output=poster_title_link)

        except:
            return render_template("error.html")


if __name__ == '__main__':
    print("App is running")
    app.run(debug=True)
