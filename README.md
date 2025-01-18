# Book Recommendation System 📚

## Overview
The **Book Recommendation System** is a web application that provides personalized book recommendations based on user preferences and collaborative filtering techniques. The application uses datasets containing information on books, users, and ratings to make intelligent suggestions.

---

## Features
- 📖 **Personalized Recommendations**: Suggest books based on user preferences and past ratings.
- 🔍 **User-Friendly Interface**: Simple and clean UI for browsing recommendations.
- 📊 **Collaborative Filtering**: Recommends books based on patterns in user behavior.
- 📂 **Dataset Integration**: Uses real-world datasets to ensure relevant recommendations.

---

## Tech Stack
- **Backend**: Python, Flask
- **Frontend**: HTML, CSS
- **Machine Learning**: Collaborative Filtering Algorithm
- **Database**: CSV-based datasets (Books, Ratings, Users)
- **Libraries**: Pandas, NumPy, Scikit-learn

---

## Installation and Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Aakhaash/Book-Recommendation-System.git
   cd Book-Recommendation-System
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open the application in your browser:
   ```
   http://127.0.0.1:5000
   ```

---

## Datasets
- **Books.csv**: Contains book metadata like titles and authors.
- **Ratings.csv**: Includes user-book interaction data (ratings).
- **Users.csv**: Stores user demographic information.

---

## Folder Structure
```
Book-Recommender-System/
│
├── app.py                 # Main application file
├── model.py               # Recommender logic
├── Datasets/              # Contains Books, Ratings, Users CSV files
├── static/                # CSS and static files
├── templates/             # HTML templates
├── __pycache__            # Dependencies
├── LICENSE                # License file
└── README.md              # Project documentation
```

---

## Future Enhancements
- Integration with larger datasets (e.g., Goodreads API).
- Enhanced recommendation algorithms (content-based filtering, deep learning models).
- User authentication for personalized profiles.

---

## License
This project is licensed under the terms of the [MIT License](LICENSE).

---
