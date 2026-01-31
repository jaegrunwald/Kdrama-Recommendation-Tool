# Korean Drama Recommendation System Guide

## 🎭 Using the Kaggle Korean Dramas Dataset

This guide shows you how to build a recommendation system using the Korean drama dataset from Kaggle.

## 📊 Dataset Challenge & Solution

**The Challenge:** The Korean drama dataset contains drama metadata (titles, genres, ratings, cast, etc.) but **does not include user watch history or ratings**.

**Our Solution:** We provide two approaches:

1. **Content-Based Recommender** - Uses drama features only (no user data needed)
2. **Simulated Hybrid Recommender** - Generates realistic user data for full hybrid system

## 🚀 Quick Start Guide

### Step 1: Download the Dataset

1. Go to: https://www.kaggle.com/datasets/saikalbatyrbekova/korean-dramas-dataset-eda
2. Download the CSV file (usually named `korean_dramas.csv` or similar)
3. Place it in the same folder as these scripts

### Step 2: Choose Your Approach

#### Option A: Content-Based Only (Simpler)

Use when you:
- Don't have user watch history
- Want quick drama-to-drama recommendations
- Need a simple solution

```bash
python kdrama_recommender.py
```

Update the filepath in the script:
```python
filepath = 'your_korean_dramas.csv'  # Change this line
```

#### Option B: Full Hybrid System (Better Recommendations)

Use when you:
- Want the best quality recommendations
- Don't mind simulated user data
- Want to explore collaborative filtering

```bash
# Step 1: Generate user data
python prepare_kdrama_data.py

# Step 2: Use hybrid recommender
python use_your_own_data.py
```

## 📋 Detailed Workflows

### Workflow 1: Content-Based Recommender

This approach recommends dramas based on similarity to dramas you liked.

```python
from kdrama_recommender import KDramaRecommender

# Initialize
recommender = KDramaRecommender()

# Load your data (update the path)
recommender.load_and_prepare_data('korean_dramas.csv')

# Prepare features
recommender.prepare_features()

# Explore the dataset
recommender.explore_dataset()

# Method 1: Find similar dramas
similar = recommender.get_similar_dramas("Crash Landing", n_recommendations=10)
print(similar)

# Method 2: Get recommendations by preferences
recs = recommender.get_recommendations_by_preferences(
    preferred_genres=['Romance', 'Comedy'],
    preferred_tags=['Strong Female Lead'],
    min_rating=8.0,
    year_range=(2018, 2023),
    n_recommendations=10
)
print(recs)
```

### Workflow 2: Simulated Hybrid System

This creates realistic user watch patterns and uses the full hybrid recommender.

```python
# STEP 1: Generate simulated data
from prepare_kdrama_data import create_hybrid_ready_dataset

watch_history, show_features = create_hybrid_ready_dataset(
    'korean_dramas.csv',
    output_prefix='kdrama'
)

# This creates:
# - kdrama_watch_history.csv (simulated user ratings)
# - kdrama_show_features.csv (drama metadata)

# STEP 2: Use with hybrid recommender
from hybrid_recommender import HybridRecommender
import pandas as pd

# Load the generated files
watch_history = pd.read_csv('kdrama_watch_history.csv')
show_features = pd.read_csv('kdrama_show_features.csv')

# Initialize and train
recommender = HybridRecommender(
    collaborative_weight=0.6,
    content_weight=0.4
)

recommender.prepare_data(watch_history, show_features)
recommender.train_collaborative_filtering(n_neighbors=20)

# Get recommendations for a user
user_id = 'user_0001'
recommendations = recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
print(recommendations)
```

## 🎨 Customization Examples

### Find Dramas Like Your Favorite

```python
from kdrama_recommender import KDramaRecommender

recommender = KDramaRecommender()
recommender.load_and_prepare_data('korean_dramas.csv')
recommender.prepare_features()

# Find dramas similar to multiple favorites
favorites = ["Goblin", "Hotel del Luna", "My Mister"]

for fav in favorites:
    print(f"\n{'='*50}")
    print(f"Dramas similar to: {fav}")
    print('='*50)
    
    similar = recommender.get_similar_dramas(fav, n_recommendations=5)
    
    for i, row in enumerate(similar.iterrows(), 1):
        idx, drama = row
        print(f"{i}. {drama['Name']} (Similarity: {drama['similarity_score']:.3f})")
```

### Create Personalized Recommendations

```python
# For a romance lover who likes recent dramas
romance_recs = recommender.get_recommendations_by_preferences(
    preferred_genres=['Romance', 'Melodrama'],
    preferred_tags=['Love Triangle', 'Strong Female Lead'],
    min_rating=8.0,
    year_range=(2020, 2024),
    n_recommendations=10
)

# For an action/thriller fan
action_recs = recommender.get_recommendations_by_preferences(
    preferred_genres=['Action', 'Thriller'],
    preferred_tags=['Crime', 'Investigation', 'Suspense'],
    min_rating=7.5,
    year_range=(2015, 2024),
    n_recommendations=10
)
```

### Simulate Different User Types

```python
from prepare_kdrama_data import UserHistorySimulator

dramas_df = pd.read_csv('korean_dramas.csv')
simulator = UserHistorySimulator(dramas_df)

# Generate more diverse users
watch_history = simulator.generate_watch_history(
    n_users=500,      # More users
    noise_level=0.4   # More randomness
)

# Save for later use
watch_history.to_csv('custom_watch_history.csv', index=False)
```

## 📊 Understanding the Korean Drama Dataset

Common columns you'll find:
- **Name/Title**: Drama name
- **Year**: Release year
- **Genre**: Drama genres (Romance, Comedy, Action, etc.)
- **Rating**: Average rating (usually 1-10 scale)
- **Number of Episodes**: Episode count
- **Cast**: Lead actors
- **Synopsis**: Plot description
- **Tags**: Descriptive tags (Strong Female Lead, Time Travel, etc.)

## 💡 Best Practices

### For Content-Based Recommender

1. **Explore First**: Run `recommender.explore_dataset()` to see available genres and tags
2. **Use Multiple Methods**: Combine similarity-based and preference-based recommendations
3. **Adjust Features**: If your dataset has different column names, update the `prepare_features()` parameters

### For Hybrid System

1. **Generate More Data**: Increase `n_users` for better collaborative filtering
2. **Tune Weights**: Experiment with different collaborative/content weight ratios
3. **Evaluate**: Use train/test splits to measure performance
4. **Regular Updates**: Regenerate with new noise_level for variety

## 🔧 Troubleshooting

### Issue: Column Not Found

**Error**: KeyError or column name errors

**Solution**: Check your dataset columns and update the script:

```python
# In kdrama_recommender.py, update the column names
recommender.prepare_features(
    name_col='YourNameColumn',
    genre_col='YourGenreColumn',
    rating_col='YourRatingColumn',
    # ... etc
)
```

### Issue: No Similar Dramas Found

**Error**: Drama name not found

**Solution**: Use partial matching or check exact spelling:

```python
# The system supports partial matching
similar = recommender.get_similar_dramas("Crash", n_recommendations=10)  # Matches "Crash Landing"

# Or view all drama names first
print(recommender.dramas_df['Name'].tolist())
```

### Issue: Recommendations Too Similar

**Solution**: Adjust the hybrid weights to favor collaborative filtering:

```python
recommender = HybridRecommender(
    collaborative_weight=0.7,  # More diversity from users
    content_weight=0.3         # Less content similarity
)
```

## 📈 Advanced Usage

### Create a Web Interface

```python
# Save this as app.py
from kdrama_recommender import KDramaRecommender

# Initialize once
recommender = KDramaRecommender()
recommender.load_and_prepare_data('korean_dramas.csv')
recommender.prepare_features()

def get_recommendations(drama_name):
    """Function to call from your web app"""
    return recommender.get_similar_dramas(drama_name, n_recommendations=10)

# Use with Flask, Streamlit, or other frameworks
```

### Batch Processing

```python
# Generate recommendations for multiple users
users_to_process = ['user_0001', 'user_0002', 'user_0003']

all_recs = {}
for user_id in users_to_process:
    recs = recommender.get_hybrid_recommendations(user_id, n_recommendations=10)
    all_recs[user_id] = recs.to_dict()

# Save to file
import json
with open('batch_recommendations.json', 'w') as f:
    json.dump(all_recs, f, indent=2)
```

## 🎯 Recommendation Quality Tips

1. **Content-Based Works Best For**:
   - New users (no history yet)
   - Finding similar dramas
   - Genre-specific searches

2. **Hybrid Works Best For**:
   - Established users (with watch history)
   - Discovering new preferences
   - Overall best recommendations

3. **Improve Results**:
   - Clean your data (handle missing values)
   - Add more features if available
   - Increase user sample size
   - Experiment with weights

## 📚 Next Steps

1. **Start Simple**: Begin with content-based recommender
2. **Test Thoroughly**: Try different dramas and preferences
3. **Add Real Data**: If you get actual user data, replace simulated data
4. **Deploy**: Build a web app or integrate into your platform
5. **Monitor**: Track which recommendations users actually watch

## 🤝 Example Integration

```python
# Simple recommendation API
class DramaRecommendationAPI:
    def __init__(self):
        self.recommender = KDramaRecommender()
        self.recommender.load_and_prepare_data('korean_dramas.csv')
        self.recommender.prepare_features()
    
    def recommend_similar(self, drama_name, count=5):
        """Get similar dramas"""
        return self.recommender.get_similar_dramas(drama_name, count)
    
    def recommend_by_mood(self, mood):
        """Recommend based on mood"""
        mood_map = {
            'romantic': {
                'genres': ['Romance'],
                'tags': ['Love', 'Love Triangle']
            },
            'thrilling': {
                'genres': ['Thriller', 'Action'],
                'tags': ['Crime', 'Mystery']
            },
            'feel-good': {
                'genres': ['Comedy', 'Slice of Life'],
                'tags': ['Friendship', 'Feel Good']
            }
        }
        
        prefs = mood_map.get(mood.lower(), mood_map['feel-good'])
        return self.recommender.get_recommendations_by_preferences(
            preferred_genres=prefs['genres'],
            preferred_tags=prefs['tags'],
            n_recommendations=10
        )

# Usage
api = DramaRecommendationAPI()
recommendations = api.recommend_by_mood('romantic')
```

---

**Happy Drama Watching! 🎬**
