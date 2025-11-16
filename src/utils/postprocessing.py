"""
Postprocessing utilities for sentiment analysis.
"""

import re
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SentimentPostprocessor:
    """
    Postprocess generated text to extract sentiment labels.
    """
    
    def __init__(self):
        self.positive_patterns = [
            r'\bpositive\b',
            r'\bpos\b', 
            r'\bgood\b',
            r'\bhappy\b',
            r'\bgreat\b',
            r'\bexcellent\b',
            r'\bamazing\b'
        ]
        
        self.negative_patterns = [
            r'\bnegative\b',
            r'\bneg\b',
            r'\bbad\b',
            r'\bsad\b',
            r'\bterrible\b',
            r'\bawful\b',
            r'\bhorrible\b'
        ]
        
        self.neutral_patterns = [
            r'\bneutral\b',
            r'\bneut\b',
            r'\bnormal\b',
            r'\bmedium\b',
            r'\baverage\b',
            r'\bok\b',
            r'\bneutral\b'
        ]
        
        # Compiled regex patterns for better performance
        self.positive_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.positive_patterns]
        self.negative_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.negative_patterns]
        self.neutral_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.neutral_patterns]

    def extract_sentiment(self, text: str) -> str:
        """
        Extract sentiment label from generated text.
        
        Args:
            text: Generated text from model
            
        Returns:
            Sentiment label: "positive", "negative", "neutral", or ""
        """
        if not text or not text.strip():
            return ""
            
        text_lower = text.lower().strip()
        
        # Check for exact matches first
        if any(pattern.search(text_lower) for pattern in self.positive_regex):
            return "positive"
        elif any(pattern.search(text_lower) for pattern in self.negative_regex):
            return "negative"
        elif any(pattern.search(text_lower) for pattern in self.neutral_regex):
            return "neutral"
        
        # Fallback: check for sentiment indicators
        return self._fallback_sentiment_analysis(text_lower)

    def _fallback_sentiment_analysis(self, text: str) -> str:
        """
        Fallback method for sentiment analysis using keyword matching.
        """
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'like', 'happy', 'positive']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'negative', 'worst']
        neutral_words = ['neutral', 'ok', 'fine', 'average', 'normal', 'medium']
        
        words = text.split()
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        neutral_count = sum(1 for word in words if word in neutral_words)
        
        if positive_count > negative_count and positive_count > neutral_count:
            return "positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            return "negative"
        elif neutral_count > positive_count and neutral_count > negative_count:
            return "neutral"
        else:
            return ""

    def batch_extract_sentiment(self, texts: List[str]) -> List[str]:
        """
        Extract sentiment labels for multiple texts.
        
        Args:
            texts: List of generated texts
            
        Returns:
            List of sentiment labels
        """
        return [self.extract_sentiment(text) for text in texts]

    def get_sentiment_confidence(self, text: str) -> Dict[str, Any]:
        """
        Get sentiment with confidence score.
        
        Args:
            text: Generated text
            
        Returns:
            Dictionary with sentiment and confidence
        """
        sentiment = self.extract_sentiment(text)
        
        if sentiment:
            return {
                "sentiment": sentiment,
                "confidence": 1.0,
                "raw_text": text
            }
        else:
            return {
                "sentiment": "",
                "confidence": 0.0,
                "raw_text": text
            }
