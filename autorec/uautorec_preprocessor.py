import numpy as np
import pandas as pd
import scipy.sparse

# AutoRec preprocessor
# Data: https://www.kaggle.com/grouplens/movielens-20m-dataset#rating.csv

USER_ID_COLUMN = 0
MOVIE_ID_COLUMN = 1
RATING_COLUMN = 2


def read_csv(pathStr):
    data = pd.read_csv(pathStr)
    data = data.to_numpy()
    return data


def create_movie2idx(data):
    movie2idx = dict()
    count = 0

    for movie in data[:, MOVIE_ID_COLUMN]:
        if movie not in movie2idx:
            movie2idx[movie] = count
            count += 1

    return movie2idx


def idx2movie(movie2idx, value):
    for k, v in movie2idx.items():
        if value == v:
            return k


def create_user_movie_matrix(data):
    # initialize matrix
    user_maxid = np.max(data[:, USER_ID_COLUMN])
    movie2idx = create_movie2idx(data)
    mat = np.zeros(shape=(user_maxid, len(movie2idx)), dtype=float)
    print('user-movie matrix = ' + str(mat.shape))

    for row in data:
        user = row[USER_ID_COLUMN]
        movie = row[MOVIE_ID_COLUMN]
        rating = row[RATING_COLUMN]
        mat[user - 1][movie2idx[movie]] = rating

    return mat

def most_common_users(mat, threshold):
    common_users = []

    for row in mat:
        num_of_ratings = np.sum(np.not_equal(row, 0))
        if num_of_ratings >= threshold:
            common_users.append(row)

    common_users = np.asarray(common_users)
    return common_users


def export(mat, output):
    # the user-movie matrix is extremely large (up to ~138k x 131k)
    # We need to export this spare matrix to an external file for further usage
    # to reduce storage space as well as increasing performance, we use spicy.sparse
    sparse_matrix = scipy.sparse.csc_matrix(mat)
    scipy.sparse.save_npz(output, sparse_matrix)


def main():
    data = read_csv('../data/movielens-20m-dataset/rating.csv')
    print('read_csv done')

    mat = create_user_movie_matrix(data)
    #mat = most_common_users(mat, threshold=400)
    print('most_common_mat: ' + str(mat.shape))
    export(mat, '../data/movielens-20m-dataset/rating.npz')

    '''
    # see sparse matrix
    sparse_matrix = scipy.sparse.load_npz('../data/movielens-20m-dataset/rating.npz')
    print(sparse_matrix)
    '''


main()
