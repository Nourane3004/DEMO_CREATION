#!/usr/bin/env python3
"""
Multi-agent Orchestrator for Trading Decisions
Agents: MACRO_INSIGHT, QUANT_MODELER, PATTERN_DETECTOR, RISK_MANAGER, RAG_AGENT
Design: dynamic orchestration, optional LLM controller for language tasks, asyncio concurrency.
Author: adapted for your project by ChatGPT
"""

import os
import json
import logging
import asyncio
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone

# LLM client placeholder (DeepSeek / other). Use env var DEEPSEEK_API_KEY or OPENROUTER_API_KEY
try:
    from openai import OpenAI  # DeepSeek uses OpenAI-compatible API
except ImportError:
    OpenAI = None  # optional; LLMService will handle absence

logger = logging.getLogger("multi_agent_orchestrator")
logging.basicConfig(level=logging.INFO)

# Try to import Alpha Vantage modules
try:
    from alpha_vantage.timeseries import TimeSeries
    from alpha_vantage.techindicators import TechIndicators
    from alpha_vantage.fundamentals import Fundamentals
    ALPHA_VANTAGE_AVAILABLE = True
except ImportError:
    ALPHA_VANTAGE_AVAILABLE = False
    logger.warning("Alpha Vantage not installed. Install with: pip install alpha-vantage")

# Import QdrantRAGAgent if available
try:
    from Trading_agent.Qdrant_RAG_agent import QdrantRAGAgent  # Adjust import path as needed
    QDRANT_RAG_AVAILABLE = True
except ImportError:
    QDRANT_RAG_AVAILABLE = False
    logger.warning("QdrantRAGAgent not available. Using simple RAG agent.")
    QdrantRAGAgent = None
# ---------------------------
# Agent interfaces (placeholders)
# Replace these minimal classes with your real agents (import them instead).
# Each agent should implement an async `run` method accepting a context dict and returning results dict.
# ---------------------------

class BaseAgent:
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

import os
import logging
from typing import Dict, Optional, Tuple
from alpha_vantage.timeseries import TimeSeries


logger = logging.getLogger("multi_agent_orchestrator")

class AlphaVantageService:
    """Service for fetching financial data from Alpha Vantage API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage service.
        
        Args:
            api_key: Your Alpha Vantage API key. If None, reads from ALPHA_V env var.
        """
        self.api_key = api_key or os.getenv("ALPHA_V")
        
        if not self.api_key:
            logger.warning("AlphaVantageService: No API key found. Please set ALPHAVANTAGE_API_KEY environment variable.")
            self.available = False
            return
        
        self.available = True
        
        # Initialize clients for different data types
        # output_format='pandas' returns data as DataFrames for easier analysis
        self.timeseries = TimeSeries(key=self.api_key, output_format='pandas')
        self.indicators = TechIndicators(key=self.api_key, output_format='pandas')
        self.fundamentals = Fundamentals(key=self.api_key)
        
        logger.info("AlphaVantageService initialized successfully")
    
    def get_stock_quote(self, symbol: str) -> Dict:
        """Get latest stock quote for a symbol."""
        if not self.available:
            return self._mock_response(f"Mock quote for {symbol}")
        
        try:
            data, meta_data = self.timeseries.get_quote_endpoint(symbol=symbol)
            return {
                'symbol': symbol,
                'price': float(data['05. price']),
                'change': float(data['09. change']),
                'change_percent': data['10. change percent'],
                'volume': int(data['06. volume']),
                'timestamp': data['07. latest trading day'],
                'success': True
            }
        except Exception as e:
            logger.error(f"Error fetching stock quote for {symbol}: {e}")
            return {'success': False, 'error': str(e), 'symbol': symbol}
    
    def get_intraday_data(self, symbol: str, interval: str = '5min', 
                          outputsize: str = 'compact') -> Tuple:
        """
        Get intraday stock data.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'IBM')
            interval: Time interval ('1min', '5min', '15min', '30min', '60min')
            outputsize: 'compact' (last 100 points) or 'full' (premium only)
        
        Returns:
            Tuple of (data, metadata)
        """
        if not self.available:
            return self._mock_intraday_data(symbol), {}
        
        try:
            data, meta_data = self.timeseries.get_intraday(
                symbol=symbol, 
                interval=interval, 
                outputsize=outputsize
            )
            return data, meta_data
        except Exception as e:
            logger.error(f"Error fetching intraday data for {symbol}: {e}")
            return None, {'error': str(e)}
    
    def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> Tuple:
        """
        Get daily historical data.
        
        Note: Free tier is limited to outputsize='compact' (last 100 days).
        'full' (20+ years) requires premium subscription[citation:9].
        """
        if not self.available:
            return self._mock_daily_data(symbol), {}
        
        try:
            data, meta_data = self.timeseries.get_daily(
                symbol=symbol, 
                outputsize=outputsize
            )
            return data, meta_data
        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            return None, {'error': str(e)}
    
    def get_technical_indicator(self, symbol: str, indicator: str = 'SMA', 
                               interval: str = 'daily', time_period: int = 20,
                               series_type: str = 'close') -> Dict:
        """
        Get technical indicator data.
        
        Args:
            symbol: Stock symbol
            indicator: Technical indicator (e.g., 'SMA', 'RSI', 'MACD', 'BBANDS')
            interval: Time interval
            time_period: Number of data points used for calculation
            series_type: 'close', 'open', 'high', 'low'
        """
        if not self.available:
            return {'data': self._mock_indicator_data(), 'success': True}
        
        try:
            # Map indicator names to method calls
            indicator_methods = {
                'SMA': self.indicators.get_sma,
                'EMA': self.indicators.get_ema,
                'RSI': self.indicators.get_rsi,
                'MACD': self.indicators.get_macd,
                'BBANDS': self.indicators.get_bbands,
                'STOCH': self.indicators.get_stoch,
                'ADX': self.indicators.get_adx,
            }
            
            if indicator not in indicator_methods:
                return {'success': False, 'error': f'Unsupported indicator: {indicator}'}
            
            method = indicator_methods[indicator]
            
            # Call appropriate method based on indicator
            if indicator in ['SMA', 'EMA', 'RSI']:
                data, meta_data = method(
                    symbol=symbol, 
                    interval=interval, 
                    time_period=time_period,
                    series_type=series_type
                )
            elif indicator == 'MACD':
                data, meta_data = method(symbol=symbol, interval=interval)
            elif indicator == 'BBANDS':
                data, meta_data = method(
                    symbol=symbol, 
                    interval=interval, 
                    time_period=time_period,
                    series_type=series_type
                )
            else:
                data, meta_data = method(symbol=symbol, interval=interval)
            
            return {'data': data, 'meta_data': meta_data, 'success': True}
            
        except Exception as e:
            logger.error(f"Error fetching {indicator} for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_fundamental_data(self, symbol: str, report_type: str = 'overview') -> Dict:
        """
        Get fundamental company data.
        
        Args:
            symbol: Stock symbol
            report_type: Type of fundamental data ('overview', 'income_statement', 
                       'balance_sheet', 'cash_flow', 'earnings')
        """
        if not self.available:
            return self._mock_fundamental_data(symbol, report_type)
        
        try:
            report_methods = {
                'overview': self.fundamentals.get_company_overview,
                'income_statement': self.fundamentals.get_income_statement_annual,
                'balance_sheet': self.fundamentals.get_balance_sheet_annual,
                'cash_flow': self.fundamentals.get_cash_flow_annual,
                'earnings': self.fundamentals.get_earnings_annual,
            }
            
            if report_type not in report_methods:
                return {'success': False, 'error': f'Unsupported report type: {report_type}'}
            
            data, meta_data = report_methods[report_type](symbol)
            return {'data': data, 'meta_data': meta_data, 'success': True}
            
        except Exception as e:
            logger.error(f"Error fetching {report_type} for {symbol}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _mock_response(self, message: str) -> Dict:
        """Return mock response when API is unavailable."""
        return {
            'symbol': 'MOCK',
            'price': 150.50,
            'change': 1.50,
            'change_percent': '1.0%',
            'volume': 1000000,
            'timestamp': '2024-01-01',
            'success': True,
            'mock': True,
            'message': message
        }
    
    def _mock_intraday_data(self, symbol: str):
        """Generate mock intraday data."""
        import pandas as pd
        import numpy as np
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='5min')
        data = pd.DataFrame({
            '1. open': np.random.uniform(145, 155, 100),
            '2. high': np.random.uniform(150, 160, 100),
            '3. low': np.random.uniform(140, 150, 100),
            '4. close': np.random.uniform(145, 155, 100),
            '5. volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)
        return data
    
    def _mock_daily_data(self, symbol: str):
        """Generate mock daily data."""
        import pandas as pd
        import numpy as np
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
        data = pd.DataFrame({
            '1. open': np.random.uniform(140, 160, 100),
            '2. high': np.random.uniform(145, 165, 100),
            '3. low': np.random.uniform(135, 155, 100),
            '4. close': np.random.uniform(140, 160, 100),
            '5. volume': np.random.randint(500000, 5000000, 100)
        }, index=dates)
        return data
    
    def _mock_indicator_data(self):
        """Generate mock indicator data."""
        import pandas as pd
        import numpy as np
        dates = pd.date_range(end=pd.Timestamp.now(), periods=50, freq='D')
        data = pd.DataFrame({
            'SMA': np.random.uniform(145, 155, 50),
            'RSI': np.random.uniform(30, 70, 50),
        }, index=dates)
        return data
    
    def _mock_fundamental_data(self, symbol: str, report_type: str) -> Dict:
        """Generate mock fundamental data."""
        mock_data = {
            'overview': {
                'Symbol': symbol,
                'Name': f'Mock Company {symbol}',
                'Description': 'A mock company for testing purposes',
                'MarketCapitalization': '1000000000',
                'PERatio': '25.5',
                'DividendYield': '1.5',
                'EPS': '5.50',
                'mock': True
            },
            'income_statement': {
                'reportedCurrency': 'USD',
                'totalRevenue': '100000000',
                'grossProfit': '60000000',
                'netIncome': '20000000',
                'mock': True
            }
        }
        
        data = mock_data.get(report_type, {'mock': True, 'type': report_type})
        return {'data': data, 'success': True, 'mock': True}

class MacroInsightAgent(BaseAgent):
    """MACRO-INSIGHT: Analyse les conditions économiques globales et les tendances macro."""
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Example: use RAG context if present to summarize
        rag_docs = context.get("rag_docs", [])
        # Placeholder logic - replace with real macro model
        await asyncio.sleep(0.1)
        return {
            "agent": "MACRO-INSIGHT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "macro_summary": "inflation easing, rates stable",
            "regime": "moderate_volatility",
            "raw_docs_used": len(rag_docs)
        }


class QuantModelerAgent(BaseAgent):
    """QUANT-MODELER: statistical models, signals, regime-aware forecasts"""
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        market_data = context.get("market_data", {})
        regime = context.get("macro_regime", "unknown")
        await asyncio.sleep(0.1)
        # Placeholder: compute a mock signal and confidence
        return {
            "agent": "QUANT-MODELER",
            "signal": "long",
            "score": 0.62,
            "regime_used": regime,
            "features_used": list(market_data.keys())[:5]
        }


class PatternDetectorAgent(BaseAgent):
    """PATTERN-DETECTOR: fast pattern detection on price/time-series"""
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        price_series = context.get("price_series", [])
        await asyncio.sleep(0.05)
        # Placeholder: detect simple momentum
        detected = {"momentum": True} if len(price_series) > 3 else {}
        return {"agent": "PATTERN-DETECTOR", "patterns": detected}


class RiskManagerAgent(BaseAgent):
    """RISK-MANAGER: aggregate risk, produce position sizing and adjustments"""
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        signals = context.get("signals", [])
        # Example risk logic: reduce size if volatility high
        await asyncio.sleep(0.05)
        exposure = 0.0
        if any(s.get("signal") == "long" for s in signals):
            exposure = 0.5
        # adjust exposure using simple rule + external stress factor
        stress = context.get("stress_multiplier", 1.0)
        return {"agent": "RISK-MANAGER", "recommended_exposure": exposure * (1.0 / stress)}


class RAGAgent(BaseAgent):
    """RAG_AGENT: retrieve relevant docs (news, reports) for a given query/context"""
    def __init__(self, vector_store=None):
        # vector_store can be an instance of Chroma/Qdrant or a simple file index
        self.vector_store = vector_store

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query", "")
        top_k = context.get("top_k", 5)
        await asyncio.sleep(0.05)
        # Placeholder: return mock docs. Replace with actual retrieval.
        docs = [
            {"id": "doc1", "text": "CPI fell 0.2% last month", "score": 0.98},
            {"id": "doc2", "text": "Central bank indicates steady rates", "score": 0.92}
        ][:top_k]
        return {"agent": "RAG_AGENT", "docs": docs}


# ---------------------------
# LLM Service helper
# ---------------------------

class LLMService:
    """Thin wrapper to call LLMs (DeepSeek). Returns content text."""
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.api_key = os.getenv("DEEPSEEK_KEY")
        if not self.api_key:
            logger.info("LLMService: no API key found in env; LLM calls will be skipped (mocked).")

    async def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # For production implement actual API call. Here we return a mocked answer if no key.
        if not self.api_key:
            logger.debug("LLMService.generate: returning mock response (no key)")
            return "MOCK_SUMMARY: " + (prompt[:200])
        # Example (synchronous) usage (adapt to chosen lib):
        # client = OpenAI(api_key=self.api_key)
        # resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=temperature)
        # return resp.choices[0].message.content
        # For safety in this template we won't call external APIs.
        return "LLM_RESPONSE_PLACEHOLDER"


# ---------------------------
# Scope Filter
# ---------------------------

class ScopeFilter:
    """Separate class for scope checking"""
    
    @staticmethod
    def is_in_scope(question: str) -> bool:
        trading_keywords = {"stock", "invest", "market", "trade", "buy", "sell", 
                           "portfolio", "risk", "asset", "etf", "fund", "price",
                           "crypto", "bitcoin", "ethereum", "forex", "commodity",
                           "bond", "option", "future", "derivative", "volatility",
                           "technical", "fundamental", "analysis", "chart", "trend"}
        question_words = set(re.findall(r'\b\w+\b', question.lower()))
        return bool(trading_keywords.intersection(question_words))
    
    @staticmethod
    def get_suggestion(question: str) -> str:
        question_lower = question.lower()
        if "retirement" in question_lower or "401k" in question_lower or "ira" in question_lower:
            return "Try asking about 'long-term investment strategies' or 'portfolio allocation'"
        elif "weather" in question_lower:
            return "Try asking about 'market conditions' or 'economic climate'"
        elif "sports" in question_lower or "game" in question_lower:
            return "Try asking about 'trading performance' or 'investment returns'"
        return "Try: 'Should I buy SPY?' or 'What's the market risk?' or 'Analyze Apple stock'"


# ---------------------------
# Simple Trading Interface
# ---------------------------

class SimpleTradingInterface:
    """User-friendly interface wrapper"""
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator or MultiAgentOrchestrator()
        self.scope_filter = ScopeFilter()
    
    async def process_question(self, user_question: str) -> Dict[str, Any]:
        """Main public method for users"""
        
        # Check scope
        if not self.scope_filter.is_in_scope(user_question):
            return {
                "type": "out_of_scope",
                "message": "I specialize in trading and market analysis. I can help with stocks, ETFs, crypto, and other financial instruments.",
                "suggestion": self.scope_filter.get_suggestion(user_question),
                "original_question": user_question,
                "success": False
            }
        
        # Build request
        request = self._build_request(user_question)
        
        # Use existing orchestrator
        result = await self.orchestrator.orchestrate(request)
        
        # Format response
        return self._format_response(result, user_question)
    
    def _build_request(self, question: str) -> Dict[str, Any]:
        """Bridge between simple questions and complex requests"""
        return {
            "type": self._detect_request_type(question),
            "query": question,
            "use_rag": True,
            "market_data": self._extract_market_data(question),
            "price_series": self._generate_price_series(question),
            "risk_params": {"max_exposure": 1.0, "stop_loss": 0.1},
            "stress_multiplier": 1.0
        }
    
    def _detect_request_type(self, question: str) -> str:
        question_lower = question.lower()
        if any(word in question_lower for word in ["buy", "sell", "should i", "recommend", "trade"]):
            return "signal_generation"
        elif any(word in question_lower for word in ["risk", "danger", "safe", "volatile", "crash"]):
            return "risk_check"
        elif any(word in question_lower for word in ["news", "outlook", "summary", "update", "today"]):
            return "daily_brief"
        return "signal_generation"  # Default
    
    def _extract_market_data(self, question: str) -> Dict[str, Any]:
        """Extract tickers/symbols from question and create market data"""
        # Simple ticker extraction
        ticker_patterns = {
            r'\b(spx|s&p|sp500)\b': 'SPX',
            r'\b(nasdaq|ndx|qqq)\b': 'NDX',
            r'\b(dow|dji)\b': 'DJI',
            r'\b(apple|aapl)\b': 'AAPL',
            r'\b(tesla|tsla)\b': 'TSLA',
            r'\b(google|googl)\b': 'GOOGL',
            r'\b(amazon|amzn)\b': 'AMZN',
            r'\b(microsoft|msft)\b': 'MSFT',
            r'\b(bitcoin|btc)\b': 'BTC',
            r'\b(ethereum|eth)\b': 'ETH',
            r'\b(gold|xau)\b': 'XAU',
            r'\b(oil|cl)\b': 'CL'
        }
        
        market_data = {}
        for pattern, ticker in ticker_patterns.items():
            if re.search(pattern, question.lower()):
                # Mock prices - in reality would fetch from API
                base_prices = {
                    'SPX': 4700, 'NDX': 16500, 'DJI': 37500,
                    'AAPL': 185, 'TSLA': 240, 'GOOGL': 140,
                    'AMZN': 155, 'MSFT': 380, 'BTC': 43000,
                    'ETH': 2300, 'XAU': 2050, 'CL': 75
                }
                price = base_prices.get(ticker, 100)
                market_data[ticker] = {
                    "price": price,
                    "vol": 0.12,  # 12% volatility
                    "change": 0.005  # 0.5% change
                }
        
        # If no specific ticker found, default to SPX
        if not market_data:
            market_data = {
                "SPX": {
                    "price": 4700,
                    "vol": 0.12,
                    "change": 0.005
                }
            }
        
        return market_data
    
    def _generate_price_series(self, question: str) -> List[float]:
        """Generate mock price series based on question"""
        # Simple mock data
        base_price = 100
        return [base_price + i * 2 + (i % 3) for i in range(20)]
    
    def _format_response(self, result: Dict[str, Any], original_question: str) -> Dict[str, Any]:
        """Format the orchestrator result for user consumption"""
        
        # Extract key information
        signals = result.get("signals", [])
        risk = result.get("risk", {})
        agent_outputs = result.get("agent_outputs", {})
        
        # Determine recommendation
        recommendation = "HOLD"
        confidence = 50
        
        if signals:
            buy_count = sum(1 for s in signals if s.get("signal") in ["long", "buy"])
            sell_count = sum(1 for s in signals if s.get("signal") in ["short", "sell"])
            
            if buy_count > sell_count:
                recommendation = "BUY"
                if signals:
                    confidence = int(signals[0].get("score", 0.5) * 100)
            elif sell_count > buy_count:
                recommendation = "SELL"
                if signals:
                    confidence = int(signals[0].get("score", 0.5) * 100)
        
        # Get position size
        position_size = risk.get("recommended_exposure", 0.5)
        position_percentage = f"{position_size * 100:.0f}%"
        
        # Risk level
        if position_size < 0.3:
            risk_level = "Low"
        elif position_size < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Generate explanation
        explanation_parts = []
        
        if "MACRO-INSIGHT" in agent_outputs:
            macro = agent_outputs["MACRO-INSIGHT"]
            explanation_parts.append(f"Macro context: {macro.get('macro_summary', 'neutral')}")
        
        if "QUANT-MODELER" in agent_outputs:
            quant = agent_outputs["QUANT-MODELER"]
            explanation_parts.append(f"Quant signal: {quant.get('signal', 'neutral')}")
        
        if "PATTERN-DETECTOR" in agent_outputs:
            pattern = agent_outputs["PATTERN-DETECTOR"]
            if pattern.get("patterns"):
                explanation_parts.append("Technical patterns detected")
        
        explanation = ". ".join(explanation_parts) if explanation_parts else "Analysis complete"
        
        # Key factors
        key_factors = []
        if "MACRO-INSIGHT" in agent_outputs:
            key_factors.append("Economic conditions")
        if "QUANT-MODELER" in agent_outputs:
            key_factors.append("Quantitative models")
        if "PATTERN-DETECTOR" in agent_outputs:
            key_factors.append("Technical patterns")
        if "RAG_AGENT" in agent_outputs:
            key_factors.append("Market news & reports")
        
        return {
            "type": "trading_analysis",
            "success": True,
            "question": original_question,
            "recommendation": recommendation,
            "confidence": confidence,
            "position_size": position_percentage,
            "risk_level": risk_level,
            "explanation": explanation,
            "key_factors": key_factors,
            "detailed_analysis": result,  # Include full analysis for debugging
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
class EnhancedTradingInterface(SimpleTradingInterface):
    """Enhanced interface with Qdrant RAG support"""
    
    def __init__(self, use_qdrant: bool = True, qdrant_config: Optional[Dict] = None):
        """
        Args:
            use_qdrant: Whether to use Qdrant RAG agent
            qdrant_config: Configuration for Qdrant (optional)
        """
        if qdrant_config is None:
            qdrant_config = {
                "collection_name": "financial_news",
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "n_results": 5,
                "use_local": False
            }
        
        if use_qdrant:
            try:
                # Initialize Qdrant RAG agent
                rag_agent = QdrantRAGAgent(**qdrant_config)
                
                # Create orchestrator with Qdrant
                orchestrator = MultiAgentOrchestrator(rag_agent=rag_agent)
                super().__init__(orchestrator)
                
                print("✅ Using Qdrant RAG agent")
            except Exception as e:
                print(f"⚠️ Qdrant initialization failed: {e}")
                print("Falling back to default setup")
                super().__init__()  # Use default SimpleTradingInterface
        else:
            # Use default SimpleTradingInterface without Qdrant
            super().__init__()
    
    def add_market_news(self, news_articles: List[Dict]):
        """Add market news to RAG knowledge base"""
        if hasattr(self.orchestrator.rag, 'batch_add_documents'):
            documents = []
            for article in news_articles:
                documents.append({
                    "text": f"{article.get('headline', '')}. {article.get('summary', '')}",
                    "metadata": {
                        "source": article.get("source", ""),
                        "date": article.get("date", ""),
                        "category": "market_news",
                        "tickers": article.get("tickers", [])
                    }
                })
            
            # Batch add to Qdrant
            self.orchestrator.rag.batch_add_documents(documents)
        else:
            print("⚠️ RAG agent doesn't support batch_add_documents")



# ---------------------------
# Orchestrator
# ---------------------------

class MultiAgentOrchestrator:
    def __init__(
        self,
        macro_agent: Optional[BaseAgent] = None,
        quant_agent: Optional[BaseAgent] = None,
        pattern_agent: Optional[BaseAgent] = None,
        risk_agent: Optional[BaseAgent] = None,
        rag_agent: Optional[BaseAgent] = None,
        llm_service: Optional[LLMService] = None
    ):
        # instantiate either given agents or defaults
        self.macro = macro_agent or MacroInsightAgent()
        self.quant = quant_agent or QuantModelerAgent()
        self.pattern = pattern_agent or PatternDetectorAgent()
        self.risk = risk_agent or RiskManagerAgent()
        self.rag = rag_agent or RAGAgent()
        self.llm = llm_service or LLMService()

        # simple cache for RAG results (in-memory)
        self._rag_cache: Dict[str, List[Dict[str, Any]]] = {}
        
    def _is_out_of_scope(self, query: str) -> bool:
        """Check if query is out of scope for trading analysis"""
        query_lower = query.lower()
        out_of_scope_keywords = [
            "retirement", "401k", "ira", "pension", "social security",
            "weather", "sports", "movie", "recipe", "cooking", "travel",
            "health", "fitness", "relationship", "dating", "politics",
            "entertainment", "gaming", "vacation", "holiday"
        ]
        return any(keyword in query_lower for keyword in out_of_scope_keywords)

    async def orchestrate(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        request example:
        {
            "type": "signal_generation",  # or "daily_brief", "backtest", "explain_signal"
            "query": "S&P500 outlook after CPI",
            "market_data": {...},
            "price_series": [...],
            "risk_params": {...},
            "use_rag": True
        }
        """
        start = datetime.now(timezone.utc)
        logger.info(f"Orchestration started for request type: {request.get('type')}")
        
        # Check if query is out of scope (optional - now handled by SimpleTradingInterface)
        query = request.get("query", "")
        if self._is_out_of_scope(query):
            logger.warning(f"Query appears out of scope: {query}")
            # Continue anyway since scope filter already caught it

        # Build base context
        context: Dict[str, Any] = {
            "market_data": request.get("market_data", {}),
            "price_series": request.get("price_series", []),
            "risk_params": request.get("risk_params", {}),
            "query": request.get("query", ""),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        # 1) Optionally run RAG first to enrich context if requested or if query is present
        if request.get("use_rag", False) or request.get("query"):
            rag_key = context["query"] or "default"
            rag_docs = await self._get_rag_docs(rag_key, top_k=request.get("rag_top_k", 5))
            context["rag_docs"] = rag_docs

            # Optionally summarize RAG docs with LLM (if available) to produce a concise macro_context
            if self.llm.api_key:
                combined_text = "\n\n".join(d["text"] for d in rag_docs)
                prompt = f"Summarize the following documents into a short macro summary and list top events:\n\n{combined_text}"
                summary = await self.llm.generate(prompt, max_tokens=256, temperature=0.1)
                context["rag_summary"] = summary
            else:
                # lightweight summary fallback
                context["rag_summary"] = " | ".join(d["text"][:200] for d in rag_docs)

        # 2) Decide dynamically which agents to run based on request type
        # Example rules:
        request_type = request.get("type", "signal_generation")
        tasks = []
        # for a signal generation we usually need macro -> quant & pattern -> risk
        if request_type == "signal_generation":
            # run macro first if RAG changed something; but macro can run concurrently and produce regime
            tasks.append(asyncio.create_task(self.macro.run(context)))
            # pattern and quant can run in parallel
            tasks.append(asyncio.create_task(self.pattern.run(context)))
            tasks.append(asyncio.create_task(self.quant.run(context)))
        elif request_type == "daily_brief":
            # prefer RAG + LLM summary + macro insights
            tasks.append(asyncio.create_task(self.rag.run(context)))
            tasks.append(asyncio.create_task(self.macro.run(context)))
        elif request_type == "risk_check":
            tasks.append(asyncio.create_task(self.risk.run(context)))
            tasks.append(asyncio.create_task(self.quant.run(context)))
        else:
            # default: run everything in parallel
            tasks.extend([
                asyncio.create_task(self.rag.run(context)),
                asyncio.create_task(self.macro.run(context)),
                asyncio.create_task(self.quant.run(context)),
                asyncio.create_task(self.pattern.run(context)),
                asyncio.create_task(self.risk.run(context))
            ])

        # Await tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Normalize outputs and handle exceptions
        agent_outputs: Dict[str, Any] = {}
        for res in results:
            if isinstance(res, Exception):
                logger.warning(f"Agent raised exception: {res}")
                continue
            if isinstance(res, dict):
                agent_name = res.get("agent") or res.get("name") or str(type(res))
                agent_outputs[agent_name] = res

        # 3) Post-process: build signals list and call risk manager to adjust exposure
        signals = []
        if "QUANT-MODELER" in agent_outputs:
            q = agent_outputs["QUANT-MODELER"]
            signals.append({"agent": "QUANT-MODELER", "signal": q.get("signal"), "score": q.get("score", 0.0)})
        # pattern detector output
        if "PATTERN-DETECTOR" in agent_outputs:
            p = agent_outputs["PATTERN-DETECTOR"]
            patterns = p.get("patterns", {})
            if patterns:
                # heuristics: pattern-derived signal
                signals.append({"agent": "PATTERN-DETECTOR", "signal": "long" if patterns.get("momentum") else "neutral", "info": patterns})

        # Some agents return non-standard keys (placeholders will have lowercase names)
        # Accept both uppercase and lowercase names (robustness)
        for key, val in agent_outputs.items():
            if isinstance(key, str) and key.lower().startswith("quant"):
                if val.get("signal"):
                    signals.append({"agent": key, "signal": val.get("signal"), "score": val.get("score", 0.0)})

        # Send aggregated signals to risk manager for sizing
        risk_context = {
            "signals": signals,
            "stress_multiplier": request.get("stress_multiplier", 1.0),
            "rag_summary": context.get("rag_summary", ""),
            "market_data": context.get("market_data", {})
        }
        risk_result = await self.risk.run(risk_context)
        agent_outputs["RISK-MANAGER"] = risk_result

        # 4) Final decision composition
        final_decision = {
            "signals": signals,
            "risk": risk_result,
            "agent_outputs": agent_outputs,
            "request_meta": {"request_type": request_type, "elapsed_seconds": (datetime.now(timezone.utc) - start).total_seconds()}
        }

        logger.info("Orchestration completed")
        return final_decision

    async def _get_rag_docs(self, key: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # Simple cache
        if key in self._rag_cache:
            return self._rag_cache[key]

        # call rag agent
        try:
            rag_resp = await self.rag.run({"query": key, "top_k": top_k})
            docs = rag_resp.get("docs", [])
            self._rag_cache[key] = docs
            return docs
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {e}")
            return []
        
# Add this AFTER the MultiAgentOrchestrator class definition in orchestrator.py

# ---------------------------
# Qdrant Integration Functions
# ---------------------------

def create_orchestrator_with_qdrant(
    use_local_qdrant: bool = False,
    collection_name: str = "trading_insights"
) -> MultiAgentOrchestrator:
    """
    Factory function to create orchestrator with Qdrant RAG agent.
    
    Args:
        use_local_qdrant: Use in-memory Qdrant for development
        collection_name: Name of the Qdrant collection
    
    Returns:
        MultiAgentOrchestrator with Qdrant RAG agent
    """
    try:
        # Try to import QdrantRAGAgent (if in separate file)
        # If QdrantRAGAgent is in the same file, just use it directly
        rag_agent = QdrantRAGAgent(
            collection_name=collection_name,
            embedding_model="BAAI/bge-small-en-v1.5",
            qdrant_url=None if use_local_qdrant else os.getenv("QDRANT_URL"),
            qdrant_api_key=None if use_local_qdrant else os.getenv("QDRANT_API_KEY"),
            use_local=use_local_qdrant
        )
    except Exception as e:
        print(f"Warning: Could not initialize QdrantRAGAgent: {e}")
        print("Falling back to default RAG agent")
        rag_agent = RAGAgent()  # Your original simple agent
    
    # Create orchestrator
    return MultiAgentOrchestrator(
        macro_agent=MacroInsightAgent(),
        quant_agent=QuantModelerAgent(),
        pattern_agent=PatternDetectorAgent(),
        risk_agent=RiskManagerAgent(),
        rag_agent=rag_agent,
        llm_service=LLMService()
    )



# ---------------------------
# Example usage
# ---------------------------

async def main():
    # Create the trading assistant
    assistant = SimpleTradingInterface()
    
    # Test questions
    questions = [
        "Should I buy Apple stock?",          # IN SCOPE
        "How to plan for retirement?",        # OUT OF SCOPE
        "What's the market risk today?",      # IN SCOPE
        "What's the weather forecast?",       # OUT OF SCOPE
        "Analyze Bitcoin price",              # IN SCOPE
        "Is now a good time to invest in tech stocks?",  # IN SCOPE
        "Give me a market summary",           # IN SCOPE
        "What's your favorite movie?",        # OUT OF SCOPE
        "Should I sell my Tesla shares?",     # IN SCOPE
        "How do I cook pasta?",               # OUT OF SCOPE
    ]
    
    print("🤖 Trading Assistant v1.0")
    print("=" * 60)
    
    for question in questions:
        print(f"\n📝 User: {question}")
        response = await assistant.process_question(question)
        
        if not response.get("success", False):
            print(f"🚫 Bot: {response.get('message', 'Out of scope')}")
            print(f"💡 Suggestion: {response.get('suggestion', 'Try a trading question')}")
        else:
            print(f"✅ Bot: Recommendation: {response['recommendation']} (Confidence: {response['confidence']}%)")
            print(f"📊 Position Size: {response['position_size']} | Risk Level: {response['risk_level']}")
            print(f"📝 Explanation: {response['explanation']}")
            print(f"🔑 Key Factors: {', '.join(response['key_factors'])}")
        
        print("-" * 60)

async def interactive_chat():
    """Interactive chat mode"""
    assistant = EnhancedTradingInterface()
    
    print("\n" + "=" * 60)
    print("🤖 Trading Assistant - Interactive Mode")
    print("Type 'quit', 'exit', or 'bye' to end the conversation")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                print("🤖 Goodbye! Happy trading!")
                break
            
            if not user_input:
                continue
            
            print("🤖 Analyzing...")
            response = await assistant.process_question(user_input)
            
            if not response.get("success", False):
                print(f"🤖 {response.get('message')}")
                if response.get('suggestion'):
                    print(f"💡 {response.get('suggestion')}")
            else:
                print(f"🤖 Recommendation: {response['recommendation']}")
                print(f"   Confidence: {response['confidence']}%")
                print(f"   Position Size: {response['position_size']}")
                print(f"   Risk Level: {response['risk_level']}")
                print(f"   📝 {response['explanation']}")
                
        except KeyboardInterrupt:
            print("\n🤖 Session ended by user.")
            break
        except Exception as e:
            print(f"🤖 Error: {e}")
            print("Please try again with a different question.")


if __name__ == "__main__":
    # Run the example
    # asyncio.run(main())
    
    # Or run interactive chat
    asyncio.run(interactive_chat())