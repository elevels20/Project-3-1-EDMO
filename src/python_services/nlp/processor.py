"""NLP processor - can be used independently or via FastAPI."""
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline as hf_pipeline
from keybert import KeyBERT
import re

class NLPProcessor:
    """Processes text for various NLP tasks."""
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        sentiment_model_name: str = "tabularisai/multilingual-sentiment-analysis",
    ):
        """Initialize NLP models.
        
        Args:
            embedding_model_name: HuggingFace model for embeddings and keywords
            sentiment_model_name: HuggingFace model for sentiment analysis
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.keyword_model = KeyBERT(embedding_model_name)
        self.sentiment_model = hf_pipeline(
            "text-classification",
            model=sentiment_model_name,
        )
    
    @staticmethod
    def preprocess_text(text: str) -> str:
        """Convert text to lowercase and remove extra spaces or punctuation.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text
    
    def extract_keywords(self, text: str, top_n: int = 5) -> Dict:
        """Extract keywords from text.
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            Dictionary with 'keywords' and 'scores' lists
        """
        cleaned = self.preprocess_text(text)
        keywords_with_scores = self.keyword_model.extract_keywords(
            cleaned,
            top_n=top_n,
        )

        if keywords_with_scores:
            keywords, scores = map(list, zip(*keywords_with_scores))
        else:
            keywords, scores = [], []

        return {
            "keywords": list(keywords),
            "scores": list(scores),
        }
    
    def analyze_sentiment(self, text: str, max_length: int = 400) -> Dict:
        """Analyze sentiment of text.
        
        Args:
            text: Input text
            max_length: Maximum chunk length to avoid token limit
            
        Returns:
            Dictionary with 'label' and 'score'
        """
        cleaned = self.preprocess_text(text)
        
        # Split text into chunks
        words = cleaned.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            # Rough estimate: 1 token ≈ 0.75 words
            if len(' '.join(current_chunk)) > max_length * 0.75:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Handle empty text
        if not chunks:
            return {"label": "neutral", "score": 0.0}
        
        # Process each chunk
        all_sentiments = {}
        for chunk in chunks:
            result = self.sentiment_model(chunk, truncation=True, max_length=512)[0]
            label = result["label"]
            score = result["score"]
            
            if label not in all_sentiments:
                all_sentiments[label] = []
            all_sentiments[label].append(score)
        
        # Average scores across chunks and find dominant sentiment
        avg_sentiments = {
            label: sum(scores) / len(scores)
            for label, scores in all_sentiments.items()
        }
        
        dominant_label = max(avg_sentiments, key=avg_sentiments.get)
        
        return {
            "label": dominant_label,
            "score": avg_sentiments[dominant_label],
        }
        
    
    def simple_nlp_features(self, text: str):
        # normalize text to lowercase for easier matching
        text_lower = text.lower()
        
        # --- 1. Pronoun Usage (Group Orientation vs. Self) ---
        # We look for "we" (collaborative) vs "I" (individual/dominant)
        # \b ensures we match "we" but not "went" or "power"
        we_count = len(re.findall(r'\b(we|us|our|ours)\b', text_lower))
        i_count = len(re.findall(r'\b(i|my|mine|me)\b', text_lower))
        
        # Calculate ratio (handling division by zero)
        if i_count == 0:
            pronoun_ratio = "Infinite (All Group)" if we_count > 0 else "Neutral"
        else:
            pronoun_ratio = round(we_count / i_count, 2)


        # --- 2. Epistemic Uncertainty (Hedging) ---
        # Words that signal openness or lack of dominance
        hedge_words = [
            "maybe", "might", "could", "think", "guess", 
            "probably", "sort of", "kind of", "don't know", "dunno"
        ]
        # Create a regex pattern that matches any of the hedge words
        hedge_pattern = r'\b(' + '|'.join(hedge_words) + r')\b'
        hedge_count = len(re.findall(hedge_pattern, text_lower))


        # --- 3. Question Types (Curiosity vs. Confirmation) ---
        # Note: This relies on punctuation or sentence structure. 
        # Since we are using regex, we look for sentences starting with specific words.
        
        # Split by sentence terminators (. ? !) to analyze sentence starts
        sentences = re.split(r'[.?!]+', text_lower)
        
        wh_questions = 0
        yes_no_questions = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence: continue
            
            # Check for Wh- words at the start of the sentence
            if re.match(r'^(who|what|where|when|why|how)\b', sentence):
                wh_questions += 1
                
            # Check for Auxiliary verbs at the start (indicates Yes/No question)
            # e.g., "Do you...", "Is it...", "Can we..."
            elif re.match(r'^(is|are|do|does|did|can|could|should|will|would)\b', sentence):
                yes_no_questions += 1

        return {
            "collaboration_ratio": pronoun_ratio,
            "hedges (uncertainty)": hedge_count,
            "wh_questions (inquiry)": wh_questions,
            "yes_no_questions (confirmation)": yes_no_questions,
            "question_count": wh_questions + yes_no_questions,
        }
    
    def embed_text(self, text: str) -> Dict:
        """Generate embedding for text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with 'embedding' and 'embedding_dim'
        """
        cleaned = self.preprocess_text(text)
        embedding = self.embedding_model.encode(cleaned).tolist()
        return {
            "embedding": embedding,
            "embedding_dim": len(embedding),
        }


# Global instance for use without FastAPI
_processor: Optional[NLPProcessor] = None


def get_processor() -> NLPProcessor:
    """Get or create the global NLP processor instance."""
    global _processor
    if _processor is None:
        _processor = NLPProcessor()
    return _processor


def preprocess_text(text: str) -> str:
    """Convenience function to preprocess text."""
    return NLPProcessor.preprocess_text(text)


def extract_keywords(text: str, processor: NLPProcessor, top_n: int = 5) -> Dict:
    """Convenience function to extract keywords."""
    return processor.extract_keywords(text, top_n)


def analyze_sentiment(text: str, processor: NLPProcessor, max_length: int = 400) -> Dict:
    """Convenience function to analyze sentiment."""
    return processor.analyze_sentiment(text, max_length)


def get_simplar_nlp_features(text: str, processor: NLPProcessor) -> Dict:
    """Convenience function to get simple NLP features."""
    return processor.simple_nlp_features(text)


def embed_text(text: str, processor: NLPProcessor) -> Dict:
    """Convenience function to generate embeddings."""
    return processor.embed_text(text)
