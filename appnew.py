import wx
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import time

# --- ML Imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# --- Configuration ---
CONTENT_W = 0.6
COLLAB_W = 0.4
TOP_N = 10
TMDB_API_KEY = "your_tmdb_api_key_here"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w200"

# Global session to speed up HTTP requests via connection pooling
http_session = requests.Session()

def fetch_poster_url(movie_title):
    try:
        params = {'api_key': TMDB_API_KEY, 'query': movie_title, 'language': 'en-US'}
        # Using the session object is faster for multiple requests
        response = http_session.get(f"{TMDB_BASE_URL}/search/movie", params=params, timeout=5)
        response.raise_for_status()
        results = response.json().get('results')
        if results and results[0].get('poster_path'):
            return f"{TMDB_IMAGE_BASE}{results[0]['poster_path']}"
    except Exception as e:
        print(f"TMDb Error for {movie_title}: {e}")
    return None

class MovieRecommenderApp(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='Hybrid Movie Recommender System', size=(1200, 800))
        self.load_all_data()
        self.init_ui()
        self.Centre()
        self.Show()

    def load_all_data(self):
        try:
            # Load and clean in one pass
            cleaner = lambda s: s.str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()
            
            self.movies_c = pd.read_csv("dataset.csv")
            self.movies_r = pd.read_csv("movies.csv")
            self.ratings = pd.read_csv("ratings.csv")
            self.movies_c = self.movies_c.drop_duplicates(subset=['title'])
            
            self.movies_c['title'] = cleaner(self.movies_c['title'])
            self.movies_r['title'] = cleaner(self.movies_r['title'])
            
            # Efficient Content Setup
            self.movies_c['tags'] = self.movies_c['overview'].fillna('') + ' ' + (self.movies_c['genre'].fillna('') + ' ') * 2
            tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
            # Keeping vector in sparse format to save memory and speed up cosine calculation
            vector = tfidf.fit_transform(self.movies_c['tags'].values.astype('U'))
            self.similarity_c = cosine_similarity(vector)
            
            # Optimized Collaborative Pivot
            self.matrix = self.ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)
            self.knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=20).fit(self.matrix.values)
            
            self.movie_list = sorted(self.movies_c['title'].unique().tolist())

        except Exception as e:
            wx.MessageBox(f"Fatal Error: {e}", 'Error', wx.OK | wx.ICON_ERROR)
            self.Destroy()

    def init_ui(self):
        BG_COLOR = wx.Colour(0, 0, 0)
        GOLD_COLOR = wx.Colour(255, 215, 0) # Define the Gold color
        
        panel = wx.Panel(self)
        panel.SetBackgroundColour(BG_COLOR)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # --- NEW TITLE SECTION ---
        # Create the title text widget
        app_title = wx.StaticText(panel, label="Movie Recommender System")
        
        # Create a bold, larger font (Size 24, Bold)
        title_font = wx.Font(24, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        app_title.SetFont(title_font)
        
        # Set the color to Gold
        app_title.SetForegroundColour(GOLD_COLOR)
        
        # Add title to the sizer with some top and bottom padding
        main_sizer.Add(app_title, 0, wx.ALIGN_CENTER | wx.TOP | wx.BOTTOM, 20)
        # -------------------------

        self.movie_choice = wx.ComboBox(panel, choices=self.movie_list, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.recommend_btn = wx.Button(panel, label="Get Recommendations")
        self.recommend_btn.Bind(wx.EVT_BUTTON, self.on_recommend)
        
        main_sizer.Add(self.movie_choice, 0, wx.ALL | wx.EXPAND, 10)
        main_sizer.Add(self.recommend_btn, 0, wx.ALIGN_CENTER | wx.BOTTOM, 10)
        
        self.scroll = wx.ScrolledWindow(panel)
        self.scroll.SetScrollRate(5, 5)
        self.scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scroll.SetSizer(self.scroll_sizer)
        
        main_sizer.Add(self.scroll, 1, wx.EXPAND | wx.ALL, 10)
        panel.SetSizer(main_sizer)
    def hybrid_recommend(self, movie_title):
        try:
            # 1. Content Scores (Vectorized lookup)
            idx_c = self.movies_c.index[self.movies_c['title'] == movie_title]
            if idx_c.empty: return "Movie not found."
            
            idx = idx_c[0]
            # Use pandas Series for easy merging later
            content_scores = pd.Series(self.similarity_c[idx], index=self.movies_c['title'])

            # 2. Collaborative Scores
            idx_r = self.movies_r.index[self.movies_r['title'] == movie_title]
            collab_scores = pd.Series(0.0, index=self.movies_c['title']) # Default 0s
            
            if not idx_r.empty:
                movie_id = self.movies_r.iloc[idx_r[0]]['movieId']
                if movie_id in self.matrix.index:
                    m_idx = self.matrix.index.get_loc(movie_id)
                    distances, indices = self.knn.kneighbors(self.matrix.values[m_idx].reshape(1, -1))
                    
                    for dist, i in zip(distances.flatten(), indices.flatten()):
                        r_movie_id = self.matrix.index[i]
                        r_title_series = self.movies_r[self.movies_r['movieId'] == r_movie_id]['title']
                        if not r_title_series.empty:
                            collab_scores[r_title_series.values[0]] = 1 - dist

            # 3. Hybrid Combination (Vectorized Math)
            # Normalize and Combine
            final_scores = (content_scores * CONTENT_W) + (collab_scores * COLLAB_W)
            
            # Remove the input movie itself
            final_scores = final_scores.drop(labels=[movie_title], errors='ignore')
            top_matches = final_scores.sort_values(ascending=False).head(TOP_N)

            # 4. Final Data Assembly
            gui_output = []
            for title, score in top_matches.items():
                details = self.movies_c[self.movies_c['title'] == title].iloc[0]
                gui_output.append({
                    'name': title,
                    'genre': details['genre'],
                    'overview': details['overview'],
                    'poster': fetch_poster_url(title),
                    'similarity': score
                })
            return gui_output

        except Exception as e:
            return f"Error: {e}"

    def on_recommend(self, event):
        movie_name = self.movie_choice.GetValue()
        if not movie_name: return
        
        self.scroll_sizer.Clear(True)
        wx.Yield()
        
        recommendations = self.hybrid_recommend(movie_name)
        if isinstance(recommendations, list):
            self.display_recommendations(recommendations)
        
        self.scroll.Layout()
        self.scroll.FitInside()

    def display_recommendations(self, recommendations):
        for movie in recommendations:
            movie_panel = wx.Panel(self.scroll)
            movie_panel.SetBackgroundColour(wx.Colour(40, 40, 40))
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            
            # Poster Download
            if movie['poster']:
                try:
                    resp = http_session.get(movie['poster'], timeout=5)
                    img = Image.open(BytesIO(resp.content)).resize((150, 225), Image.Resampling.LANCZOS)
                    wx_img = wx.Bitmap.FromBuffer(150, 225, img.convert("RGB").tobytes())
                    sizer.Add(wx.StaticBitmap(movie_panel, bitmap=wx_img), 0, wx.ALL, 5)
                except:
                    sizer.Add(wx.StaticText(movie_panel, label="No Image"), 0, wx.ALL, 40)
            
            # Info
            info_sizer = wx.BoxSizer(wx.VERTICAL)
            t = wx.StaticText(movie_panel, label=f"{movie['name']} (Score: {movie['similarity']:.3f})")
            t.SetForegroundColour(wx.Colour(255, 215, 0))
            info_sizer.Add(t, 0, wx.ALL, 5)

           
            overview_text = str(movie['overview']) if pd.notna(movie['overview']) else "No description available."
            desc = wx.StaticText(movie_panel, label=overview_text)
            desc.SetForegroundColour(wx.WHITE)
            desc.Wrap(600)
            info_sizer.Add(desc, 0, wx.ALL, 5)
            
            sizer.Add(info_sizer, 1, wx.EXPAND | wx.ALL, 5)
            movie_panel.SetSizer(sizer)
            self.scroll_sizer.Add(movie_panel, 0, wx.EXPAND | wx.ALL, 5)

if __name__ == '__main__':
    app = wx.App()
    MovieRecommenderApp()
    app.MainLoop()