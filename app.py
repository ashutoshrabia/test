from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = "/app/cache"

app = FastAPI(
    title="News Article Similarity Search API",
    description="API for finding similar news articles",
    version="1.0.0"
)

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class DocumentStore:
    def __init__(self, csv_path: str = "Articles.csv"):
        self.model = None
        self.dimension = 384  
        self.index = None
        self.documents = []
        self.csv_path = csv_path
        self.load_csv()

    def load_csv(self):
        logger.info(f"Looking for the CSV at: {self.csv_path}")
        if not os.path.exists(self.csv_path):
            logger.error(f"Couldn’t find the CSV file at {self.csv_path}")
            raise FileNotFoundError(f"CSV file not found at {self.csv_path}")

        logger.info(f"Loading the CSV from {self.csv_path}")
        df = pd.read_csv(self.csv_path, encoding='latin1')
        logger.info(f"Found these columns in the CSV: {df.columns.tolist()}")

        required_columns = ['Article', 'Date', 'Heading', 'NewsType']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing some required columns. Found: {df.columns.tolist()}")
            raise ValueError("CSV needs Article, Date, Heading, and NewsType columns!")

        # Grabbing all the articles as a list of strings
        articles = df['Article'].astype(str).tolist()
        logger.info(f"Loaded {len(articles)} articles from the CSV")

        # Loading the model and turning the articles into embeddings
        logger.info("Loading the SentenceTransformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Turning articles into embeddings...")
        embeddings = self.model.encode(articles, show_progress_bar=True)

        # Setting up FAISS to do fast similarity searches
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        logger.info(f"Added {len(embeddings)} embeddings to the FAISS index")

        for idx, row in df.iterrows():
            self.documents.append({
                'id': idx,
                'article': str(row['Article']),
                'date': str(row['Date']),
                'heading': str(row['Heading']),
                'news_type': str(row['NewsType'])
            })
        logger.info(f"Saved {len(self.documents)} documents for later")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        # Making sure everything’s ready before we search
        if not self.model or not self.index:
            logger.error("Something’s wrong—DocumentStore isn’t set up right")
            raise RuntimeError("DocumentStore isn’t ready. Missing model or index.")

        logger.info(f"Searching for: {query}, returning top {top_k} results")
        query_embedding = self.model.encode([query])[0]

        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        results = []

        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                similarity = 1 - (dist / 2) 
                results.append({
                    'id': doc['id'],
                    'article': doc['article'],
                    'date': doc['date'],
                    'heading': doc['heading'],
                    'news_type': doc['news_type'],
                    'similarity': float(similarity)
                })
        logger.info(f"Got {len(results)} results for the query: {query}")
        return results

doc_store = DocumentStore()

# Some basic CSS 
def get_styles():
    return """
    <style>
        body { font-family: Arial, sans-serif; background-color: #f4f4f4; text-align: center; }
        .container { max-width: 800px; margin: 50px auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); }
        h1 { color: #333; }
        a { text-decoration: none; color: #007BFF; }
        a:hover { text-decoration: underline; }
        ul { list-style-type: none; padding: 0; }
        li { background: #fff; margin: 10px 0; padding: 15px; border-radius: 5px; box-shadow: 0 0 5px rgba(0, 0, 0, 0.1); text-align: left; }
        .search-box { margin: 20px 0; }
        input[type="text"], input[type="number"] { padding: 10px; width: 70%; border: 1px solid #ccc; border-radius: 5px; }
        input[type="submit"] { padding: 10px 20px; border: none; background: #007BFF; color: white; border-radius: 5px; cursor: pointer; }
        input[type="submit"]:hover { background: #0056b3; }
    </style>
    """

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the app...")
    if not doc_store.documents:
        logger.error("Uh oh, DocumentStore didn’t load any documents")
        raise RuntimeError("DocumentStore didn’t load properly")
    logger.info("App started up fine!")

# The main page 
@app.get("/", response_class=HTMLResponse)
async def root():
    return f"""
    <html>
        <head><title>News Article Similarity Search</title>{get_styles()}</head>
        <body>
            <div class="container">
                <h1>Welcome to News Article Similarity Search API</h1>
                <p>Go to <a href="/search">search</a> to try a search.</p>
                <p>Or use the <a href="/docs">API Docs</a> for POST requests.</p>
            </div>
        </body>
    </html>
    """

@app.get("/search", response_class=HTMLResponse)
async def search_form(request: Request, query: str = "", top_k: int = 5):
    if not query:  
        return f"""
        <html>
            <head><title>Search Articles</title>{get_styles()}</head>
            <body>
                <div class="container">
                    <h1>Search Articles</h1>
                    <form method="get" class="search-box">
                        <input type="text" name="query" placeholder="Enter search query" required>
                        <input type="number" name="top_k" value="5" min="1" max="10">
                        <input type="submit" value="Search">
                    </form>
                </div>
            </body>
        </html>
        """

    results = doc_store.search(query, top_k)
    html_content = f"""
    <html>
        <head><title>Search Results</title>{get_styles()}</head>
        <body>
            <div class="container">
                <h1>Search Results for "{query}"</h1>
                <p><a href="/search">New Search</a></p>
                <ul>
    """
    for result in results:
        html_content += f"""
            <li>
                <strong>{result['heading']}</strong> ({result['date']}) - {result['news_type']}<br>
                <em>Similarity: {result['similarity']:.2f}</em><br>
                {result['article'][:200]}...
            </li>
        """
    html_content += """</ul></div></body></html>"""
    return HTMLResponse(content=html_content)

@app.post("/api/search")
async def search_documents(query: SearchQuery):
    results = doc_store.search(query=query.query, top_k=query.top_k)
    return {"results": results}

@app.get("/api/search")
async def search_documents_get(q: str, top_k: int = 5):
    results = doc_store.search(query=q, top_k=top_k)
    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)