"""
Korean Drama Recommendation System
Specialized for the Kaggle Korean Dramas Dataset

This script handles the Korean dramas dataset which contains drama metadata
but NO user ratings/watch history. We'll create:
1. Content-based recommendations using drama features
2. Simulation of user preferences for hybrid approach
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class KDramaRecommender:
    """
    Content-based recommendation system for Korean dramas
    """
    
    def __init__(self):
        self.dramas_df = None
        self.content_similarity = None
        self.tfidf_vectorizer = None
        self.scaler = MinMaxScaler()
        
    def load_and_prepare_data(self, filepath):
        """
        Load the Korean drama dataset and prepare features
        
        Expected columns (adjust based on your actual dataset):
        - Name/Title
        - Year
        - Genre
        - Rating
        - Number of Episodes
        - Cast
        - Synopsis/Description
        - Tags
        """
        print("Loading Korean drama dataset...")
        self.dramas_df = pd.read_csv(filepath)
        
        print(f"\nDataset Info:")
        print(f"- Total dramas: {len(self.dramas_df)}")
        print(f"\nColumns found: {list(self.dramas_df.columns)}")
        print(f"\nFirst few rows:")
        print(self.dramas_df.head())
        
        # Show data types and missing values
        print(f"\nData quality:")
        print(self.dramas_df.info())
        
        return self.dramas_df
    
    def prepare_features(self, 
                        name_col='Name',
                        genre_col='Genre', 
                        year_col='Year',
                        rating_col='Rating',
                        episodes_col='Number of Episodes',
                        cast_col='Cast',
                        synopsis_col='Synopsis',
                        tags_col='Tags'):
        """
        Prepare features for content-based filtering
        
        Args:
            *_col: Column names in your dataset (adjust as needed)
        """
        print("\n" + "="*60)
        print("Preparing Features...")
        print("="*60)
        
        df = self.dramas_df.copy()
        
        # Create a unique ID if not present
        if 'drama_id' not in df.columns:
            df['drama_id'] = df.index
        
        # Handle missing values
        text_columns = [genre_col, cast_col, synopsis_col, tags_col]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Combine text features for TF-IDF
        print("Creating combined text features...")
        df['combined_features'] = ''
        
        # Add genre (most important for dramas)
        if genre_col in df.columns:
            df['combined_features'] += df[genre_col] + ' '
        
        # Add tags (very important for K-dramas)
        if tags_col in df.columns:
            df['combined_features'] += df[tags_col] + ' '
        
        # Add year decade (group similar eras)
        if year_col in df.columns:
            df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
            df['decade'] = (df[year_col] // 10) * 10
            df['combined_features'] += 'decade_' + df['decade'].astype(str) + ' '
        
        # Add cast (popular actors)
        if cast_col in df.columns:
            df['combined_features'] += df[cast_col] + ' '
        
        # Calculate TF-IDF similarity based on text features
        print("Calculating content similarity...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df['combined_features'])
        
        # Calculate cosine similarity
        self.content_similarity = cosine_similarity(tfidf_matrix)
        
        # Store prepared dataframe
        self.dramas_df = df
        
        print(f"[OK] Content similarity matrix shape: {self.content_similarity.shape}")
        print(f"[OK] Feature preparation complete!")
        
    def get_similar_dramas(self, drama_name, n_recommendations=10):
        """
        Get dramas similar to a given drama based on content
        
        Args:
            drama_name: Name of the drama (partial match supported)
            n_recommendations: Number of recommendations to return
        """
        # Find the drama (case-insensitive partial match)
        name_col = self._find_name_column()
        mask = self.dramas_df[name_col].str.lower().str.contains(drama_name.lower(), na=False)
        
        if not mask.any():
            print(f"Drama '{drama_name}' not found!")
            print("\nDid you mean one of these?")
            suggestions = self.dramas_df[name_col].sample(min(5, len(self.dramas_df)))
            for i, drama in enumerate(suggestions, 1):
                print(f"{i}. {drama}")
            return pd.DataFrame()
        
        # Get the index of the drama
        drama_idx = self.dramas_df[mask].index[0]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity[drama_idx]))
        
        # Sort by similarity (excluding the drama itself)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        
        # Get drama indices
        drama_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]
        
        # Return recommendations with scores
        recommendations = self.dramas_df.iloc[drama_indices].copy()
        recommendations['similarity_score'] = scores
        
        return recommendations
    
    def get_recommendations_by_preferences(self, 
                                          preferred_genres=None,
                                          preferred_tags=None,
                                          min_rating=None,
                                          year_range=None,
                                          n_recommendations=10):
        """
        Get recommendations based on user preferences
        
        Args:
            preferred_genres: List of genres (e.g., ['Romance', 'Comedy'])
            preferred_tags: List of tags (e.g., ['Strong Female Lead'])
            min_rating: Minimum rating (e.g., 8.0)
            year_range: Tuple of (min_year, max_year) (e.g., (2015, 2023))
            n_recommendations: Number of recommendations
        """
        df = self.dramas_df.copy()
        
        # Find column names
        genre_col = self._find_genre_column()
        rating_col = self._find_rating_column()
        year_col = self._find_year_column()
        tags_col = self._find_tags_column()
        
        # Create a scoring system
        df['preference_score'] = 0.0
        
        # Genre matching
        if preferred_genres and genre_col:
            for genre in preferred_genres:
                genre_match = df[genre_col].str.contains(genre, case=False, na=False)
                df.loc[genre_match, 'preference_score'] += 2.0
        
        # Tag matching
        if preferred_tags and tags_col:
            for tag in preferred_tags:
                tag_match = df[tags_col].str.contains(tag, case=False, na=False)
                df.loc[tag_match, 'preference_score'] += 1.5
        
        # Rating filter
        if min_rating and rating_col:
            df.loc[df[rating_col] >= min_rating, 'preference_score'] += 1.0
        
        # Year range filter
        if year_range and year_col:
            min_year, max_year = year_range
            in_range = (df[year_col] >= min_year) & (df[year_col] <= max_year)
            df.loc[in_range, 'preference_score'] += 0.5
        
        # Filter out dramas with score = 0
        df = df[df['preference_score'] > 0]
        
        # Sort by preference score and rating
        if rating_col:
            df = df.sort_values(['preference_score', rating_col], ascending=[False, False])
        else:
            df = df.sort_values('preference_score', ascending=False)
        
        return df.head(n_recommendations)
    
    def _find_name_column(self):
        """Helper to find the name/title column"""
        possible_names = ['Name', 'name', 'Title', 'title', 'Drama', 'drama', 'drama_name']
        for col in possible_names:
            if col in self.dramas_df.columns:
                return col
        return self.dramas_df.columns[0]  # Default to first column
    
    def _find_genre_column(self):
        """Helper to find genre column"""
        possible = ['Genre', 'genre', 'Genres', 'genres']
        for col in possible:
            if col in self.dramas_df.columns:
                return col
        return None
    
    def _find_rating_column(self):
        """Helper to find rating column"""
        possible = ['Rating', 'rating', 'Score', 'score', 'rank']
        for col in possible:
            if col in self.dramas_df.columns:
                return col
        return None
    
    def _find_year_column(self):
        """Helper to find year column"""
        possible = ['Year', 'year', 'Release Year', 'release_year']
        for col in possible:
            if col in self.dramas_df.columns:
                return col
        return None
    
    def _find_tags_column(self):
        """Helper to find tags column"""
        possible = ['Tags', 'tags', 'Tag', 'tag']
        for col in possible:
            if col in self.dramas_df.columns:
                return col
        return None
    
    def explore_dataset(self):
        """
        Explore the dataset to understand available genres, tags, etc.
        """
        print("\n" + "="*60)
        print("Dataset Exploration")
        print("="*60)
        
        genre_col = self._find_genre_column()
        tags_col = self._find_tags_column()
        rating_col = self._find_rating_column()
        year_col = self._find_year_column()
        
        # Show genre distribution
        if genre_col:
            print(f"\nGenre Distribution:")
            genres = self.dramas_df[genre_col].str.split(',').explode()
            genres = genres.str.strip()
            print(genres.value_counts().head(10))
        
        # Show popular tags
        if tags_col:
            print(f"\nPopular Tags:")
            tags = self.dramas_df[tags_col].str.split(',').explode()
            tags = tags.str.strip()
            print(tags.value_counts().head(10))
        
        # Show rating distribution
        if rating_col:
            print(f"\nRating Statistics:")
            print(self.dramas_df[rating_col].describe())
        
        # Show year distribution
        if year_col:
            print(f"\nYear Range:")
            print(f"Oldest: {self.dramas_df[year_col].min()}")
            print(f"Newest: {self.dramas_df[year_col].max()}")


def example_usage():
    """
    Example of how to use the KDrama recommender
    """
    print("="*60)
    print("Korean Drama Recommendation System")
    print("="*60)
    
    # Initialize recommender
    recommender = KDramaRecommender()
    
    # STEP 1: Load your dataset
    # Replace with your actual file path
    filepath = 'kdrama_DATASET.csv'
    
    try:
        recommender.load_and_prepare_data(filepath)
    except FileNotFoundError:
        print(f"\n[!] File not found: {filepath}")
        print("\nInstructions:")
        print("1. Download the dataset from Kaggle")
        print("2. Update the 'filepath' variable above with your file path")
        print("3. Run this script again")
        return
    
    # STEP 2: Prepare features
    # You may need to adjust column names based on your dataset
    recommender.prepare_features()
    
    # STEP 3: Explore the dataset
    recommender.explore_dataset()
    
    # STEP 4: Get recommendations by similarity
    print("\n" + "="*60)
    print("Example 1: Find Similar Dramas")
    print("="*60)
    
    # Example: Find dramas similar to a popular one
    similar = recommender.get_similar_dramas("Crash Landing", n_recommendations=5)
    
    if not similar.empty:
        name_col = recommender._find_name_column()
        rating_col = recommender._find_rating_column()
        genre_col = recommender._find_genre_column()
        
        print(f"\nDramas similar to 'Crash Landing on You':")
        for i, row in enumerate(similar.iterrows(), 1):
            idx, drama = row
            print(f"\n{i}. {drama[name_col]}")
            if rating_col:
                print(f"   Rating: {drama[rating_col]}")
            if genre_col:
                print(f"   Genre: {drama[genre_col]}")
            print(f"   Similarity: {drama['similarity_score']:.3f}")
    
    # STEP 5: Get recommendations by preferences
    print("\n" + "="*60)
    print("Example 2: Recommendations by Preferences")
    print("="*60)
    
    preferences = recommender.get_recommendations_by_preferences(
        preferred_genres=['Romance', 'Comedy'],
        preferred_tags=['Strong Female Lead'],
        min_rating=8.0,
        year_range=(2015, 2023),
        n_recommendations=5
    )
    
    print("\nRecommended dramas based on preferences:")
    print("(Romance/Comedy, Strong Female Lead, Rating >= 8.0, 2015-2023)")
    
    name_col = recommender._find_name_column()
    rating_col = recommender._find_rating_column()
    
    for i, row in enumerate(preferences.iterrows(), 1):
        idx, drama = row
        print(f"\n{i}. {drama[name_col]}")
        if rating_col:
            print(f"   Rating: {drama[rating_col]}")
        print(f"   Preference Score: {drama['preference_score']:.1f}")


if __name__ == "__main__":
    example_usage()
