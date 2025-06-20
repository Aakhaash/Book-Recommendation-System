import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets with samples
books = pd.read_csv('./Datasets/Books.csv',low_memory=False)
ratings = pd.read_csv('./Datasets/Ratings.csv',low_memory=False).sample(n=700000,random_state=42)
users = pd.read_csv('./Datasets/Users.csv',low_memory=False).sample(n=150000,random_state=42)

# print(f"Books shape: {books.shape}")
# print(f"Ratings shape: {ratings.shape}")
# print(f"Users shape: {users.shape}")

# Popularity Based Recommender System
ratings_with_name = ratings.merge(books, on='ISBN')
print(f"Ratings with names shape: {ratings_with_name.shape}")

num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num-ratings'}, inplace=True)

avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg-rating'}, inplace=True)

popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')

# Adjust the threshold based on your data - start with lower values
min_ratings_threshold = 250
print(f"Using minimum ratings threshold: {min_ratings_threshold}")

popular_df = popular_df[popular_df['num-ratings'] >= min_ratings_threshold].sort_values('avg-rating', ascending=False).head(50)
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num-ratings','avg-rating']]

print(f"Popular books found: {len(popular_df)}")

# Collaborative Filtering Based Recommender System
print("Building collaborative filtering model...")

# Use more lenient thresholds for sampled data
user_rating_threshold = max(10, ratings_with_name.groupby('User-ID').count()['Book-Rating'].quantile(0.7))  # 70th percentile
book_rating_threshold = max(5, ratings_with_name.groupby('Book-Title').count()['Book-Rating'].quantile(0.7))   # 70th percentile

print(f"User rating threshold: {user_rating_threshold}")
print(f"Book rating threshold: {book_rating_threshold}")

# Filter users who have rated enough books
x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] >= user_rating_threshold
considered_users = x[x].index
print(f"Considered users: {len(considered_users)}")

if len(considered_users) == 0:
    print("Warning: No users meet the criteria. Using top 100 most active users.")
    user_activity = ratings_with_name.groupby('User-ID').count()['Book-Rating'].sort_values(ascending=False)
    considered_users = user_activity.head(100).index

filtered_ratings = ratings_with_name[ratings_with_name['User-ID'].isin(considered_users)]
print(f"Filtered ratings shape: {filtered_ratings.shape}")

# Filter books that have been rated by enough users
y = filtered_ratings.groupby('Book-Title').count()['Book-Rating'] >= book_rating_threshold
famous_books = y[y].index
print(f"Famous books: {len(famous_books)}")

if len(famous_books) == 0:
    print("Warning: No books meet the criteria. Using top 500 most rated books.")
    book_popularity = filtered_ratings.groupby('Book-Title').count()['Book-Rating'].sort_values(ascending=False)
    famous_books = book_popularity.head(500).index

final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]
print(f"Final ratings shape: {final_ratings.shape}")

# Create pivot table
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

print(f"Pivot table shape: {pt.shape}")

# Check if pivot table has sufficient data
if pt.shape[0] == 0 or pt.shape[1] == 0:
    print("Error: Pivot table is empty. Cannot build similarity matrix.")
    similarity_score = None
    pt = None
else:
    print("Computing similarity scores...")
    similarity_score = cosine_similarity(pt)
    print(f"Similarity matrix shape: {similarity_score.shape}")

# Function to recommend books
def recommend(book_name):
    """
    Recommend books based on collaborative filtering.
    Returns list of recommended book titles.
    """
    if pt is None or similarity_score is None:
        print("Collaborative filtering model not available. Returning popular books instead.")
        # Fallback to popular books
        if len(popular_df) > 0:
            return popular_df.head(5)['Book-Title'].tolist()
        else:
            return ["No recommendations available"]
    
    try:
        # Check if book exists in our pivot table
        if book_name not in pt.index:
            print(f"Book '{book_name}' not found in database.")
            # Find closest match
            available_books = pt.index.tolist()
            close_matches = [book for book in available_books if book_name.lower() in book.lower()]
            if close_matches:
                print(f"Did you mean: {close_matches[:3]}")
                book_name = close_matches[0]  # Use first close match
            else:
                print("Returning popular books instead.")
                return popular_df.head(5)['Book-Title'].tolist()
        
        # Get book index
        index = np.where(pt.index == book_name)[0][0]
        
        # Calculate similar books
        similar_books = sorted(list(enumerate(similarity_score[index])), 
                             key=lambda x: x[1], reverse=True)[1:6]
        
        recommended_books = []
        for i in similar_books:
            recommended_books.append(pt.index[i[0]])
        
        return recommended_books
        
    except Exception as e:
        print(f"Error in recommendation: {str(e)}")
        # Fallback to popular books
        return popular_df.head(5)['Book-Title'].tolist()

# Test the recommendation function
if pt is not None and len(pt) > 0:
    print(f"\nAvailable books for testing: {pt.index[:5].tolist()}")
    test_book = pt.index[0]
    print(f"Testing recommendation for: {test_book}")
    test_recommendations = recommend(test_book)
    print(f"Recommendations: {test_recommendations}")