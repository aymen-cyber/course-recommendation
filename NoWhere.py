
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel, euclidean_distances
from math import sqrt, pow, exp
import spacy
nlp = spacy.load("en_core_web_sm")





def squared_sum(x):
  """ return 3 rounded square rooted value """
  return round(sqrt(sum([a*a for a in x])),3)
 
def euclidean_distance(x,y):
  """ return euclidean distance between two lists """
  return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))


def distance_to_similarity(distance):
  return 1/exp(distance)


def jaccard_similarity(x,y):
  """ returns the jaccard similarity between two lists """
  intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
  union_cardinality = len(set.union(*[set(x), set(y)]))
  return intersection_cardinality/float(union_cardinality)

df = pd.read_csv("Combined_Clean_Courses.csv")

def fdf():
  # df1 = df.drop(['Unnamed: 0'], axis=1)Email
  df1 = df
  return df1


# Example data
data = ["This is a sentence", "This is another sentence"]

# Create a CountVectorizer object
#count_vect = CountVectorizer()

# Fit the CountVectorizer object to the data
#cv_mat = count_vect.fit_transform(data)

# Create a DataFrame of the count matrix with column names
#df_cv_words = pd.DataFrame(cv_mat.toarray(),columns=count_vect.get_feature_names())



count_vect = CountVectorizer()
#cv_mat = count_vect.fit_transform(df['Hard_Skills'].values.astype('U'))
cv_mat = count_vect.fit_transform(df['clean_course_title'].values.astype('U'))
#print(cv_mat)

#df_cv_words = pd.DataFrame(cv_mat.todense(),columns=count_vect.get_feature_names())
df_cv_words = pd.DataFrame(cv_mat.toarray(),columns=count_vect.get_feature_names_out())

cosine_sim_mat = cosine_similarity(cv_mat)
#course_indices = pd.Series(df.index,index=df['Hard_Skills']).drop_duplicates()
course_indices = pd.Series(df.index,index=df['clean_course_title']).drop_duplicates()


def recommend_course(title,num_of_rec=10):
    # ID for title
    idx = course_indices[title]
    if idx is not None:
        #idx = course_indices.get(title, "Title not found")
        # Course Indice
        # Search inside cosine_sim_mat
        scores = list(enumerate(cosine_sim_mat[idx]))
        # Scores
        # Sort Scores
        sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)
        # Recomm
        selected_course_indices = [i[0] for i in sorted_scores[1:]]
        selected_course_scores = [i[1] for i in sorted_scores[1:]]
        result = df['clean_course_title'].iloc[selected_course_indices]
        rec_df = pd.DataFrame(result)
        rec_df['similarity_scores'] = selected_course_scores
    else:
        # Handle the case when the course title is not found
        return f"Course '{title}' not found in the dataset."
    return rec_df.head(num_of_rec)



euclidean_distances_mat = euclidean_distances(cv_mat)
course_indices_euc = pd.Series(df.index,index=df['clean_course_title']).drop_duplicates()


def recommend_course_euc(title,num_of_rec=10):
    # ID for title
    idx = course_indices[title]
    # Course Indice
    # Search inside cosine_sim_mat
    scores = list(enumerate(euclidean_distances_mat[idx]))
    # Scores
    # Sort Scores
    sorted_scores = sorted(scores,key=lambda x:x[1],reverse=True)
    # Recomm
    selected_course_indices = [i[0] for i in sorted_scores[1:]]
    selected_course_scores = [i[1] for i in sorted_scores[1:]]
    result = df['clean_course_title'].iloc[selected_course_indices]
    rec_df = pd.DataFrame(result)
    rec_df['similarity_scores'] = selected_course_scores
    return rec_df.head(num_of_rec)
