"""
Simulate User Watch History for Korean Drama Dataset

Since the Kaggle dataset doesn't include user ratings/watch history,
this script creates realistic simulated user data so you can use the
full hybrid recommendation system.
"""

import pandas as pd
import numpy as np
from kdrama_recommender import KDramaRecommender


class UserHistorySimulator:
    """
    Generate realistic user watch history from drama metadata
    """
    
    def __init__(self, dramas_df):
        self.dramas_df = dramas_df
        
    def simulate_user_preferences(self, n_users=200):
        """
        Create user preference profiles based on common patterns
        """
        np.random.seed(42)
        
        # Define user archetypes (realistic K-drama viewer types)
        archetypes = {
            'romance_lover': {
                'genres': ['Romance', 'Melodrama'],
                'tags': ['Strong Female Lead', 'Love Triangle'],
                'rating_bias': 0.5,  # Tends to rate higher
                'watch_count_range': (15, 30)
            },
            'action_fan': {
                'genres': ['Action', 'Thriller'],
                'tags': ['Crime', 'Investigation'],
                'rating_bias': 0.3,
                'watch_count_range': (10, 25)
            },
            'comedy_enthusiast': {
                'genres': ['Comedy', 'Slice of Life'],
                'tags': ['Friendship', 'Feel Good'],
                'rating_bias': 0.4,
                'watch_count_range': (12, 28)
            },
            'historical_buff': {
                'genres': ['Historical', 'Period'],
                'tags': ['Royalty', 'Politics'],
                'rating_bias': 0.2,
                'watch_count_range': (8, 20)
            },
            'diverse_viewer': {
                'genres': None,  # Watches everything
                'tags': None,
                'rating_bias': 0.0,
                'watch_count_range': (20, 40)
            }
        }
        
        # Assign archetypes to users
        user_archetypes = np.random.choice(
            list(archetypes.keys()),
            size=n_users,
            p=[0.3, 0.2, 0.25, 0.15, 0.1]  # Distribution of user types
        )
        
        return user_archetypes, archetypes
    
    def generate_watch_history(self, n_users=200, noise_level=0.2):
        """
        Generate realistic watch history
        
        Args:
            n_users: Number of users to simulate
            noise_level: How much randomness (0-1, higher = more random)
        """
        print(f"Generating watch history for {n_users} users...")
        
        user_archetypes, archetypes = self.simulate_user_preferences(n_users)
        
        # Find column names
        recommender = KDramaRecommender()
        recommender.dramas_df = self.dramas_df
        
        name_col = recommender._find_name_column()
        genre_col = recommender._find_genre_column()
        rating_col = recommender._find_rating_column()
        tags_col = recommender._find_tags_column()
        
        # Create drama IDs
        if 'drama_id' not in self.dramas_df.columns:
            self.dramas_df['drama_id'] = [f"drama_{i:04d}" for i in range(len(self.dramas_df))]
        
        watch_history = []
        
        for user_id in range(n_users):
            archetype_name = user_archetypes[user_id]
            archetype = archetypes[archetype_name]
            
            # Determine how many dramas this user watches
            min_watch, max_watch = archetype['watch_count_range']
            n_watches = np.random.randint(min_watch, max_watch + 1)
            
            # Filter dramas based on user preference
            if archetype['genres'] is not None:
                # User has genre preferences
                preferred_dramas = self.dramas_df.copy()
                
                if genre_col:
                    # Create a mask for preferred genres
                    genre_mask = preferred_dramas[genre_col].str.contains(
                        '|'.join(archetype['genres']),
                        case=False,
                        na=False
                    )
                    
                    # 70% chance to pick from preferred, 30% random
                    if np.random.random() < 0.7:
                        candidate_dramas = preferred_dramas[genre_mask]
                    else:
                        candidate_dramas = preferred_dramas
                else:
                    candidate_dramas = preferred_dramas
            else:
                # Diverse viewer - watches everything
                candidate_dramas = self.dramas_df.copy()
            
            # Ensure we have enough dramas
            if len(candidate_dramas) < n_watches:
                candidate_dramas = self.dramas_df.copy()
            
            # Sample dramas for this user
            watched_dramas = candidate_dramas.sample(
                n=min(n_watches, len(candidate_dramas)),
                replace=False
            )
            
            # Generate ratings for each watched drama
            for _, drama in watched_dramas.iterrows():
                # Base rating from drama's actual rating
                if rating_col and pd.notna(drama[rating_col]):
                    base_rating = drama[rating_col]
                    # Convert to 1-5 scale if needed
                    if base_rating > 5:
                        base_rating = base_rating / 2
                else:
                    base_rating = 4.0  # Default
                
                # Add user's archetype bias
                user_rating = base_rating + archetype['rating_bias']
                
                # Add some noise
                user_rating += np.random.normal(0, noise_level)
                
                # Clip to 1-5 range
                user_rating = np.clip(user_rating, 1, 5)
                
                # Round to nearest 0.5
                user_rating = round(user_rating * 2) / 2
                
                watch_history.append({
                    'user_id': f'user_{user_id:04d}',
                    'drama_id': drama['drama_id'],
                    'drama_name': drama[name_col],
                    'rating': user_rating,
                    'user_type': archetype_name
                })
        
        watch_history_df = pd.DataFrame(watch_history)
        
        print(f"✓ Generated {len(watch_history_df)} watch records")
        print(f"✓ {watch_history_df['user_id'].nunique()} unique users")
        print(f"✓ {watch_history_df['drama_id'].nunique()} unique dramas watched")
        print(f"\nRating distribution:")
        print(watch_history_df['rating'].value_counts().sort_index())
        
        return watch_history_df


def create_hybrid_ready_dataset(kdrama_filepath, output_prefix='kdrama'):
    """
    Create watch history and show features files ready for hybrid recommender
    
    Args:
        kdrama_filepath: Path to the Korean drama CSV file
        output_prefix: Prefix for output files
    """
    print("="*60)
    print("Creating Hybrid-Ready Dataset from Korean Drama Data")
    print("="*60)
    
    # Load the drama dataset
    print("\n1. Loading drama dataset...")
    dramas_df = pd.read_csv(kdrama_filepath)
    print(f"✓ Loaded {len(dramas_df)} dramas")
    
    # Find column names dynamically
    recommender = KDramaRecommender()
    recommender.dramas_df = dramas_df
    
    name_col = recommender._find_name_column()
    genre_col = recommender._find_genre_column()
    rating_col = recommender._find_rating_column()
    year_col = recommender._find_year_column()
    
    # Create drama IDs
    if 'drama_id' not in dramas_df.columns:
        dramas_df['drama_id'] = [f"drama_{i:04d}" for i in range(len(dramas_df))]
    
    # Generate watch history
    print("\n2. Generating simulated user watch history...")
    simulator = UserHistorySimulator(dramas_df)
    watch_history = simulator.generate_watch_history(n_users=200, noise_level=0.3)
    
    # Create show features file
    print("\n3. Creating show features file...")
    show_features = dramas_df[['drama_id', name_col]].copy()
    
    # Add available features
    if genre_col:
        show_features['genre'] = dramas_df[genre_col]
    if rating_col:
        show_features['avg_rating'] = dramas_df[rating_col]
    if year_col:
        show_features['year'] = dramas_df[year_col]
    
    # Add number of episodes if available
    episodes_cols = ['Number of Episodes', 'Episodes', 'episodes', 'num_episodes']
    for col in episodes_cols:
        if col in dramas_df.columns:
            show_features['num_episodes'] = dramas_df[col]
            break
    
    # Prepare watch history for hybrid recommender
    watch_history_clean = watch_history[['user_id', 'drama_id', 'rating']].copy()
    
    # Rename for compatibility
    watch_history_clean.columns = ['user_id', 'show_id', 'rating']
    show_features = show_features.rename(columns={'drama_id': 'show_id'})
    
    # Save files
    print("\n4. Saving files...")
    watch_history_file = f'{output_prefix}_watch_history.csv'
    features_file = f'{output_prefix}_show_features.csv'
    
    watch_history_clean.to_csv(watch_history_file, index=False)
    show_features.to_csv(features_file, index=False)
    
    print(f"✓ Saved watch history to: {watch_history_file}")
    print(f"✓ Saved show features to: {features_file}")
    
    # Display sample data
    print("\n" + "="*60)
    print("Sample Data Preview")
    print("="*60)
    
    print("\nWatch History (first 10 rows):")
    print(watch_history_clean.head(10))
    
    print("\nShow Features (first 10 rows):")
    print(show_features.head(10))
    
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total users: {watch_history_clean['user_id'].nunique()}")
    print(f"Total shows: {watch_history_clean['show_id'].nunique()}")
    print(f"Total ratings: {len(watch_history_clean)}")
    print(f"Average ratings per user: {len(watch_history_clean) / watch_history_clean['user_id'].nunique():.1f}")
    print(f"Average ratings per show: {len(watch_history_clean) / watch_history_clean['show_id'].nunique():.1f}")
    
    print("\n✅ Dataset ready for hybrid recommendation system!")
    print(f"\nNext steps:")
    print(f"1. Use these files with the hybrid_recommender.py")
    print(f"2. Load: watch_history = pd.read_csv('{watch_history_file}')")
    print(f"3. Load: show_features = pd.read_csv('{features_file}')")
    
    return watch_history_clean, show_features


if __name__ == "__main__":
    # Example usage
    print("Korean Drama Dataset Preparation Tool")
    print("\nThis script will:")
    print("1. Load your Korean drama dataset")
    print("2. Generate realistic simulated user watch history")
    print("3. Create files compatible with the hybrid recommender")
    
    # UPDATE THIS PATH to your Korean drama CSV file
    kdrama_file = 'korean_dramas.csv'
    
    try:
        watch_history, show_features = create_hybrid_ready_dataset(
            kdrama_file,
            output_prefix='kdrama'
        )
    except FileNotFoundError:
        print(f"\n❌ File not found: {kdrama_file}")
        print("\n📝 Instructions:")
        print("1. Download the dataset from Kaggle:")
        print("   https://www.kaggle.com/datasets/saikalbatyrbekova/korean-dramas-dataset-eda")
        print("2. Update the 'kdrama_file' variable in this script")
        print("3. Run this script again")
