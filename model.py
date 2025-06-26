import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

print("Starting model initialization...")

# Simple data loading without complex type optimization
def load_data_simple():
    """Simple data loading approach"""
    try:
        print("Loading Books.csv...")
        books = pd.read_csv('./Datasets/Books.csv', encoding='latin-1', low_memory=False)
        print(f"Books loaded: {books.shape}")
        
        print("Loading Ratings.csv...")
        ratings = pd.read_csv('./Datasets/Ratings.csv', encoding='latin-1')
        print(f"Ratings loaded: {ratings.shape}")
        
        print("Loading Users.csv...")
        users = pd.read_csv('./Datasets/Users.csv', encoding='latin-1')
        print(f"Users loaded: {users.shape}")
        
        return books, ratings, users
    except Exception as e:
        print(f"Error in simple loading: {e}")
        return None, None, None

# Check if preprocessed data exists
pickle_files = ['popular_books.pkl', 'similarity_matrix.pkl', 'pivot_table.pkl', 'books_data.pkl']
all_exist = all(os.path.exists(f) for f in pickle_files)

if all_exist:
    print("Loading preprocessed data...")
    try:
        with open('popular_books.pkl', 'rb') as f:
            popular_df = pickle.load(f)
        with open('similarity_matrix.pkl', 'rb') as f:
            similarity_score = pickle.load(f)
        with open('pivot_table.pkl', 'rb') as f:
            pt = pickle.load(f)
        with open('books_data.pkl', 'rb') as f:
            books = pickle.load(f)
        print("✅ Preprocessed data loaded successfully!")
    except Exception as e:
        print(f"Error loading preprocessed data: {e}")
        all_exist = False

if not all_exist:
    print("Processing data from CSV files...")
    books, ratings, users = load_data_simple()
    
    if books is None:
        print("❌ Failed to load CSV files. Please check:")
        print("1. Files exist in ./Datasets/ folder")
        print("2. Files are not corrupted") 
        print("3. Try running test_data.py first")
        raise Exception("Failed to load datasets")
    
    print("✅ CSV files loaded successfully!")
    print("🔄 Building recommendation system...")
    
    # Popularity Based Recommender System
    print("Building popularity-based recommendations...")
    ratings_with_name = ratings.merge(books, on='ISBN')
    
    # Free up memory
    del ratings, users
    
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num-ratings'}, inplace=True)
    
    avg_rating_df = ratings_with_name.groupby('Book-Title').mean(numeric_only=True)['Book-Rating'].reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg-rating'}, inplace=True)
    
    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    popular_df = popular_df[popular_df['num-ratings'] >= 250].sort_values('avg-rating', ascending=False).head(50)
    
    popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
        ['Book-Title', 'Book-Author', 'Image-URL-M', 'num-ratings', 'avg-rating']
    ]
    
    print(f"Popular books found: {len(popular_df)}")
    
    # Collaborative Filtering (simplified for memory)
    print("Building collaborative filtering...")
    
    # Use smaller thresholds for local testing
    x = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 50  # Reduced threshold
    considered_users = x[x].index
    print(f"Considered users: {len(considered_users)}")
    
    filtered_ratings = ratings_with_name[ratings_with_name['User-ID'].isin(considered_users)]
    
    y = filtered_ratings.groupby('Book-Title').count()['Book-Rating'] >= 20  # Reduced threshold
    famous_books = y[y].index
    print(f"Famous books: {len(famous_books)}")
    
    final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]
    
    # Free up memory
    del ratings_with_name, filtered_ratings
    
    # Limit size for memory efficiency
    if len(final_ratings) > 100000:  # Limit to 100k ratings
        final_ratings = final_ratings.sample(100000, random_state=42)
        print("Limited to 100k ratings for memory efficiency")
    
    print("Creating pivot table...")
    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)
    
    print(f"Pivot table shape: {pt.shape}")
    
    # Limit books for similarity calculation
    if len(pt) > 500:  # Limit to 500 books
        pt = pt.head(500)
        print("Limited to 500 books for similarity calculation")
    
    print("Computing similarity matrix...")
    similarity_score = cosine_similarity(pt)
    print(f"Similarity matrix shape: {similarity_score.shape}")
    
    # Save the processed data
    print("Saving processed data...")
    try:
        with open('popular_books.pkl', 'wb') as f:
            pickle.dump(popular_df, f)
        with open('similarity_matrix.pkl', 'wb') as f:
            pickle.dump(similarity_score, f)
        with open('pivot_table.pkl', 'wb') as f:
            pickle.dump(pt, f)
        with open('books_data.pkl', 'wb') as f:
            pickle.dump(books, f)
        print("✅ Data saved successfully!")
    except Exception as e:
        print(f"Warning: Could not save processed data: {e}")

print("✅ Model initialization complete!")

def recommend(book_name):
    """Function to recommend books"""
    try:
        if book_name not in pt.index:
            # Try to find partial matches
            matches = [book for book in pt.index if book_name.strip().lower() in book.strip().lower()]
            if matches:
                book_name = matches[0]
                print(f"Using closest match: {book_name}")
            else:
                print(f"Book '{book_name}' not found in database")
                return []
        
        index = np.where(pt.index == book_name)[0][0]
        similar_books = sorted(list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True)[1:6]
        
        recommended_books = []
        for i in similar_books:
            recommended_books.append(pt.index[i[0]])
        
        return recommended_books
    except Exception as e:
        print(f"Error in recommendation: {e}")
        return []

# Test the recommendation function
if __name__ == "__main__":
    print("\n" + "="*50)
    print("TESTING RECOMMENDATION SYSTEM")
    print("="*50)
    
    print(f"\nAvailable books for recommendation: {len(pt)}")
    print("Sample books:")
    for i, book in enumerate(pt.index[:5]):
        print(f"{i+1}. {book}")
    
    # Test with first book
    if len(pt) > 0:
        test_book = pt.index[0]
        print(f"\nTesting recommendation for: {test_book}")
        recommendations = recommend(test_book)
        print(f"Recommendations: {recommendations}")
    else:
        print("No books available for testing!")