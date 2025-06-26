from flask import Flask, render_template, request
import pandas as pd
from model import popular_df, recommend, books
import os

app = Flask(__name__)

@app.route('/')
def index():
    try:
        return render_template('index.html', books=popular_df.to_dict('records'))
    except Exception as e:
        print(f"Error in index route: {e}")
        return render_template('index.html', books=[])

@app.route('/recommend', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        try:
            book_name = request.form['book_name']
            recommended_book_names = recommend(book_name)
            
            if not recommended_book_names:
                return render_template('recommend.html', books=[], error="No recommendations found for this book.")
            
            recommended_books = []
            for book in recommended_book_names:
                try:
                    book_matches = books[books['Book-Title'] == book]
                    if not book_matches.empty:
                        book_info = book_matches.iloc[0]
                        recommended_books.append({
                            'Book-Title': book_info['Book-Title'],
                            'Book-Author': book_info['Book-Author'],
                            'Image-URL-M': book_info['Image-URL-M'],
                            'ISBN': book_info['ISBN']
                        })
                except Exception as e:
                    print(f"Error processing book {book}: {e}")
                    continue
            
            return render_template('recommend.html', books=recommended_books)
        except Exception as e:
            print(f"Error in recommendation: {e}")
            return render_template('recommend.html', books=[], error="An error occurred while getting recommendations.")
    
    return render_template('recommend.html')

@app.route('/health')
def health_check():
    return {"status": "healthy"}, 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # For production deployment with Gunicorn
    port = int(os.environ.get('PORT', 5000))