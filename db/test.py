from crewai import Agent, Task, Crew, LLM
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict
import json
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON, ForeignKey, Text, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Create SQLAlchemy Base
Base = declarative_base()

# Database Models
class DocumentPage(Base):
    __tablename__ = 'document_pages'
    
    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True)
    title = Column(String)
    chunks = relationship("ContentChunk", back_populates="page")
    created_at = Column(DateTime, default=datetime.utcnow)

class ContentChunk(Base):
    __tablename__ = 'content_chunks'
    
    id = Column(Integer, primary_key=True)
    page_id = Column(Integer, ForeignKey('document_pages.id'))
    content = Column(Text)
    vector = Column(ARRAY(Float))  # Store TF-IDF vector
    page = relationship("DocumentPage", back_populates="chunks")
    created_at = Column(DateTime, default=datetime.utcnow)

class ChatHistory(Base):
    __tablename__ = 'chat_history'
    
    id = Column(Integer, primary_key=True)
    question = Column(Text)
    answer = Column(Text)
    sources = Column(JSON)  # Store relevant source URLs and titles
    timestamp = Column(DateTime, default=datetime.utcnow)

class DocumentationCrawler:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.visited_urls = set()
        self.content_store = {}
        
    def crawl(self, url: str):
        if url in self.visited_urls or not url.startswith(self.base_url):
            return
        
        try:
            print(f"Crawling: {url}")
            self.visited_urls.add(url)
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            main_content = soup.find('div', {'class': 'document'}) or soup.find('main')
            content = main_content.get_text(strip=True) if main_content else ''
            
            links = [a['href'] for a in soup.find_all('a', href=True)]
            absolute_links = []
            for link in links:
                if link.startswith('/'):
                    link = f"{self.base_url.rstrip('/')}{link}"
                if link.startswith(self.base_url):
                    absolute_links.append(link)
            
            chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
            self.content_store[url] = {
                'chunks': chunks,
                'title': soup.title.string if soup.title else url
            }
            
            for link in absolute_links:
                self.crawl(link)
                
        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")

class DocumentationChatbot:
    def __init__(self, base_url: str, db_url: str):
        self.llm = LLM(
            model="ollama/llama3.2:1b",
            base_url="http://localhost:11434"
        )
        
        self.crawler = DocumentationCrawler(base_url)
        self.vectorizer = None
        self.base_url = base_url
        
        # Database setup
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def create_agents(self):
        return Agent(
            role='Documentation Expert',
            goal='Answer questions about documentation accurately',
            backstory='I am an expert at understanding and explaining documentation',
            llm=self.llm
        )
    
    def initialize_knowledge_base(self):
        """Initialize or load knowledge base from PostgreSQL"""
        # Check if we already have content for this URL
        existing_pages = self.session.query(DocumentPage).filter_by(url=self.base_url).first()
        
        if existing_pages:
            print("Loading existing knowledge base from database...")
            self.vectorizer = TfidfVectorizer()
            chunks = self.session.query(ContentChunk).all()
            self.vectorizer.fit([chunk.content for chunk in chunks])
        else:
            print("Building new knowledge base...")
            self.crawler.crawl(self.base_url)
            self._store_crawled_content()
    
    def _store_crawled_content(self):
        """Store crawled content in PostgreSQL"""
        # Prepare content for vectorization
        all_chunks = []
        for url, data in self.crawler.content_store.items():
            # Create document page
            page = DocumentPage(url=url, title=data['title'])
            self.session.add(page)
            self.session.flush()  # Get the page ID
            
            # Add chunks
            for chunk in data['chunks']:
                all_chunks.append(chunk)
                content_chunk = ContentChunk(
                    page_id=page.id,
                    content=chunk
                )
                self.session.add(content_chunk)
        
        # Compute vectors
        self.vectorizer = TfidfVectorizer()
        vectors = self.vectorizer.fit_transform(all_chunks)
        
        # Update chunks with vectors
        chunks = self.session.query(ContentChunk).all()
        for chunk, vector in zip(chunks, vectors.toarray()):
            chunk.vector = vector.tolist()
        
        self.session.commit()
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        query_vector = self.vectorizer.transform([query]).toarray()[0]
        
        # Get all chunks and compute similarities
        chunks = self.session.query(ContentChunk).all()
        similarities = [np.dot(chunk.vector, query_vector) for chunk in chunks]
        
        # Get top k results
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            chunk = chunks[idx]
            results.append({
                'content': chunk.content,
                'url': chunk.page.url,
                'title': chunk.page.title,
                'relevance_score': float(similarities[idx])
            })
        
        return results
    
    def generate_response(self, query: str, search_results: List[Dict]) -> str:
        # Create context from search results
        context = "\n\n".join([
            f"From {result['title']}:\n{result['content']}"
            for result in search_results if result['relevance_score'] > 0.1
        ])
        
        # Get recent chat history
        recent_history = self.session.query(ChatHistory)\
            .order_by(ChatHistory.timestamp.desc())\
            .limit(5)\
            .all()
        
        history_text = "\n".join([
            f"Human: {entry.question}\nAssistant: {entry.answer}"
            for entry in reversed(recent_history)
        ])
        
        # Create task for the agent
        task = Task(
            description=f"""
            Answer the following question using the provided context.
            Question: {query}
            
            Context:
            {context}
            
            Previous conversation:
            {history_text}
            """,
            expected_output="A detailed answer to the user's question based on the provided context.",
            agent=self.create_agents()
        )
        
        # Get response from agent
        crew = Crew(
            agents=[task.agent],
            tasks=[task]
        )
        
        try:
            response = crew.kickoff()
            return response
        except Exception as e:
            return f"I apologize, but I couldn't generate a proper response. This might be because I couldn't find relevant information in the documentation. Could you please rephrase your question or ask about a different topic?"
    
    def _save_chat_entry(self, question: str, answer: str, sources: List[Dict]):
        """Save chat entry to PostgreSQL"""
        chat_entry = ChatHistory(
            question=question,
            answer=answer,
            sources=[{'url': s['url'], 'title': s['title']} for s in sources[:3]]
        )
        self.session.add(chat_entry)
        self.session.commit()
    
    def chat_loop(self):
        print("\nWelcome to the Documentation Chatbot!")
        print("Ask any questions about the documentation. Type 'exit' to end the conversation.")
        print("\nNote: I'm currently processing documentation from:", self.base_url)
        
        while True:
            try:
                query = input("\nYou: ").strip()
                
                if query.lower() == 'exit':
                    print("\nGoodbye!")
                    break
                
                if not query:
                    continue
                
                # Search relevant content
                results = self.search(query)
                
                if not any(result['relevance_score'] > 0.1 for result in results):
                    print("\nI couldn't find any relevant information in the documentation.")
                    continue
                
                # Generate response
                response = self.generate_response(query, results)
                
                # Save to database
                self._save_chat_entry(query, response, results)
                
                # Print response with source references
                print("\nAssistant:", response)
                print("\nSources:")
                for result in results[:3]:
                    if result['relevance_score'] > 0.1:
                        print(f"- {result['title']}: {result['url']}")
            
            except KeyboardInterrupt:
                print("\n\nInterrupted by user.")
                break
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again with a different question.")

def main():
    # Get documentation URL from user
    default_url = "https://thecatapi.com/"
    url = input(f"Enter documentation URL (press Enter for default: {default_url}): ").strip()
    url = url if url else default_url
    
    # Database connection string - UPDATE THESE VALUES
    db_url = "postgresql://aa_nadim:aa_nadim123@localhost:5432/crewai_db"
    
    # Initialize chatbot
    chatbot = DocumentationChatbot(url, db_url)
    
    # Initialize knowledge base
    print("\nPreparing knowledge base...")
    chatbot.initialize_knowledge_base()
    
    # Start chat loop
    chatbot.chat_loop()

if __name__ == "__main__":
    main()