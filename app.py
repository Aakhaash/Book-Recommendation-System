from flask import Flask, render_template, request
import pandas as pd
from model import popular_df, recommend, books
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', books=popular_df.to_dict('records'))

@app.route('/recommend', methods=['GET', 'POST'])
def recommendation():
    if request.method == 'POST':
        book_name = request.form['book_name']
        recommended_book_names = recommend(book_name)
        recommended_books = []
        for book in recommended_book_names:
            book_info = books[books['Book-Title'] == book].iloc[0]
            recommended_books.append({
                'Book-Title': book_info['Book-Title'],
                'Book-Author': book_info['Book-Author'],
                'Image-URL-M': book_info['Image-URL-M'],
                'ISBN': book_info['ISBN']
            })
        return render_template('recommend.html', books=recommended_books)
    return render_template('recommend.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)