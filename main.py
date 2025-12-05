"""
main.py - Trading Assistant System Entry Point
Integrates Qdrant RAG agent, multi-agent orchestrator, and user interface.
"""


from dotenv import load_dotenv

load_dotenv()

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from orchestrator import (
        BaseAgent,
        MacroInsightAgent,
        QuantModelerAgent,
        PatternDetectorAgent,
        RiskManagerAgent,
        QdrantRAGAgent,  # Your Qdrant agent
        LLMService,
        MultiAgentOrchestrator,
        SimpleTradingInterface,
        EnhancedTradingInterface,
        create_orchestrator_with_qdrant,
        ScopeFilter
    )
    print("✅ Successfully imported all orchestrator components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure all components are defined in orchestrator.py")
    sys.exit(1)


class TradingAssistantSystem:
    """Main system class that coordinates all components"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the complete trading assistant system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = self._load_config(config_path)
        self.scope_filter = ScopeFilter()
        self.assistant = None
        
        print("=" * 60)
        print("🤖 Trading Assistant System v2.0")
        print("=" * 60)
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "use_qdrant": True,
            "qdrant_local": True,
            "qdrant_collection": "trading_knowledge",
            "embedding_model": "all-MiniLM-L6-v2",
            "max_results": 5,
            "interactive_mode": True,
            "log_level": "INFO",
            "sample_data": True
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
                print(f"📄 Loaded configuration from {config_path}")
            except Exception as e:
                print(f"⚠️ Could not load config file: {e}")
        
        # Override with environment variables
        env_config = {
            "use_qdrant": os.getenv("USE_QDRANT", "").lower() not in ["false", "0", "no"],
            "qdrant_local": os.getenv("QDRANT_LOCAL", "").lower() not in ["false", "0", "no"],
            "qdrant_url": os.getenv("QDRANT_URL"),
            "qdrant_api_key": os.getenv("QDRANT_API_KEY")
        }
        
        default_config.update({k: v for k, v in env_config.items() if v is not None})
        return default_config
    
    def _setup_qdrant_agent(self) -> QdrantRAGAgent:
        """Initialize and configure Qdrant RAG agent"""
        print("\n🔧 Setting up Qdrant RAG agent...")
        
        qdrant_config = {
            "collection_name": self.config.get("qdrant_collection", "trading_knowledge"),
            "embedding_model_name": self.config.get("embedding_model", "all-MiniLM-L6-v2"),
            "use_local": self.config.get("qdrant_local", True),
            "n_results": self.config.get("max_results", 5),
            "device": "cpu"  # Change to "cuda" if you have GPU
        }
        
        # Add cloud configuration if not using local
        if not qdrant_config["use_local"]:
            qdrant_config.update({
                "qdrant_url": self.config.get("qdrant_url"),
                "qdrant_api_key": self.config.get("qdrant_api_key")
            })
        
        try:
            rag_agent = QdrantRAGAgent(**qdrant_config)
            
            # Load sample data if configured
            if self.config.get("sample_data", True):
                print("📥 Loading sample financial data...")
                rag_agent.load_sample_financial_data()
            
            return rag_agent
            
        except Exception as e:
            print(f"❌ Failed to initialize Qdrant: {e}")
            print("Falling back to simple RAG agent...")
            # Return a simple agent if Qdrant fails
            from orchestrator import RAGAgent
            return RAGAgent()
    
    def _create_assistant(self) -> SimpleTradingInterface:
        """Create the trading assistant with appropriate configuration"""
        use_qdrant = self.config.get("use_qdrant", True)
        
        if use_qdrant:
            try:
                # Create EnhancedTradingInterface with Qdrant
                print("🚀 Creating enhanced assistant with Qdrant RAG...")
                
                assistant = EnhancedTradingInterface(
                    use_qdrant=True,
                    qdrant_config={
                        "collection_name": self.config.get("qdrant_collection", "trading_knowledge"),
                        "use_local": self.config.get("qdrant_local", True),
                        "n_results": self.config.get("max_results", 5)
                    }
                )
                
                print("✅ Enhanced assistant created")
                return assistant
                
            except Exception as e:
                print(f"⚠️ Enhanced assistant failed: {e}")
                print("Falling back to simple assistant...")
        
        # Fallback to simple assistant
        print("🔧 Creating simple assistant...")
        assistant = SimpleTradingInterface()
        print("✅ Simple assistant created")
        return assistant
    
    def _display_banner(self):
        """Display system information"""
        print("\n" + "=" * 60)
        print("SYSTEM STATUS")
        print("=" * 60)
        print(f"• Qdrant RAG: {'ENABLED' if self.config.get('use_qdrant') else 'DISABLED'}")
        print(f"• Qdrant Mode: {'LOCAL' if self.config.get('qdrant_local') else 'CLOUD'}")
        print(f"• Collection: {self.config.get('qdrant_collection', 'trading_knowledge')}")
        print(f"• Embedding Model: {self.config.get('embedding_model', 'all-MiniLM-L6-v2')}")
        print(f"• Scope Filtering: ENABLED")
        print("=" * 60)
    
    async def run_example_queries(self):
        """Run a series of example queries to demonstrate the system"""
        examples = [
            {
                "question": "Should I buy Apple stock?",
                "description": "Signal generation with specific ticker"
            },
            {
                "question": "What's the latest Fed interest rate news?",
                "description": "RAG-based market news retrieval"
            },
            {
                "question": "How risky is the market right now?",
                "description": "Risk assessment query"
            },
            {
                "question": "Give me a market summary",
                "description": "Daily brief request"
            },
            {
                "question": "How to plan for retirement?",
                "description": "Out of scope - should be filtered"
            },
            {
                "question": "What's the weather forecast?",
                "description": "Out of scope - should be filtered"
            }
        ]
        
        print("\n🧪 RUNNING EXAMPLE QUERIES")
        print("=" * 60)
        
        for i, example in enumerate(examples, 1):
            print(f"\n{i}. {example['description']}")
            print(f"   Q: {example['question']}")
            
            response = await self.assistant.process_question(example["question"])
            
            if not response.get("success", False):
                print(f"   🚫 {response.get('message', 'Out of scope')}")
                if response.get('suggestion'):
                    print(f"   💡 {response.get('suggestion')}")
            else:
                print(f"   ✅ Recommendation: {response['recommendation']}")
                print(f"   📊 Confidence: {response['confidence']}%")
                print(f"   ⚖️ Position: {response['position_size']} | Risk: {response['risk_level']}")
                
                # Show RAG info if available
                if response.get('detailed_analysis', {}).get('agent_outputs', {}).get('RAG_AGENT'):
                    rag_info = response['detailed_analysis']['agent_outputs']['RAG_AGENT']
                    if rag_info.get('docs'):
                        print(f"   📄 RAG retrieved {len(rag_info['docs'])} documents")
                        if rag_info['docs']:
                            print(f"   Top document: {rag_info['docs'][0]['text'][:100]}...")
    
    async def run_interactive_mode(self):
        """Run interactive chat mode"""
        print("\n💬 INTERACTIVE MODE")
        print("=" * 60)
        print("Type your questions about trading, stocks, or market analysis.")
        print("Commands:")
        print("  • 'rag test' - Test RAG retrieval")
        print("  • 'stats' - Show system statistics")
        print("  • 'config' - Show current configuration")
        print("  • 'quit', 'exit', 'bye' - End session")
        print("=" * 60)
        
        conversation_history = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                # Check for commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\n🤖 Goodbye! Happy trading! 📈")
                    break
                
                elif user_input.lower() == 'rag test':
                    await self._handle_rag_test()
                    continue
                
                elif user_input.lower() == 'stats':
                    self._display_system_stats()
                    continue
                
                elif user_input.lower() == 'config':
                    self._display_config()
                    continue
                
                elif user_input.lower() == 'clear':
                    conversation_history = []
                    print("🗑️ Conversation cleared")
                    continue
                
                elif user_input.lower() == 'help':
                    self._display_help()
                    continue
                
                # Process trading question
                print("🤖 Analyzing...", end='', flush=True)
                
                # Add to history
                conversation_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Process the question
                response = await self.assistant.process_question(user_input)
                
                # Add response to history
                conversation_history.append({
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Clear the "Analyzing..." text
                print("\r" + " " * 50 + "\r", end='')
                
                # Display response
                if not response.get("success", False):
                    print(f"🤖 {response.get('message', 'I cannot answer that question.')}")
                    if response.get('suggestion'):
                        print(f"💡 {response.get('suggestion')}")
                else:
                    print(f"🤖 Recommendation: {response['recommendation']}")
                    print(f"   Confidence: {response['confidence']}%")
                    print(f"   Position: {response['position_size']} | Risk: {response['risk_level']}")
                    print(f"   📝 {response['explanation']}")
                    
                    # Optionally show more details
                    if response.get('key_factors'):
                        print(f"   🔑 Key factors: {', '.join(response['key_factors'])}")
                    
                    # Show RAG info
                    if response.get('detailed_analysis', {}).get('agent_outputs', {}).get('RAG_AGENT'):
                        rag_info = response['detailed_analysis']['agent_outputs']['RAG_AGENT']
                        if rag_info.get('docs'):
                            print(f"   📄 Retrieved {len(rag_info['docs'])} relevant documents")
                
            except KeyboardInterrupt:
                print("\n\n🤖 Session interrupted.")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")
    
    async def _handle_rag_test(self):
        """Handle RAG testing command"""
        print("\n🔍 RAG TEST MODE")
        print("Enter a query to test RAG retrieval, or 'back' to return:")
        
        while True:
            test_query = input("Test query: ").strip()
            
            if test_query.lower() in ['back', 'return', 'b']:
                break
            
            if not test_query:
                continue
            
            try:
                # Direct RAG query if available
                if hasattr(self.assistant.orchestrator.rag, 'similarity_search'):
                    print("Searching...")
                    results = self.assistant.orchestrator.rag.similarity_search(test_query, top_k=3)
                    
                    if not results:
                        print("   No documents found.")
                    else:
                        print(f"   Found {len(results)} documents:")
                        for i, doc in enumerate(results, 1):
                            print(f"   {i}. Score: {doc['score']:.3f}")
                            print(f"      Source: {doc.get('source', 'unknown')}")
                            print(f"      Text: {doc['text'][:150]}...")
                            print()
                else:
                    print("   RAG agent doesn't support direct search.")
                    
            except Exception as e:
                print(f"   Error: {e}")
    
    def _display_system_stats(self):
        """Display system statistics"""
        print("\n📊 SYSTEM STATISTICS")
        print("=" * 40)
        
        if hasattr(self.assistant.orchestrator.rag, 'get_collection_stats'):
            try:
                stats = self.assistant.orchestrator.rag.get_collection_stats()
                print(f"Collection: {stats.get('collection_name', 'unknown')}")
                print(f"Vectors count: {stats.get('vectors_count', 0)}")
                print(f"Points count: {stats.get('points_count', 0)}")
                print(f"Status: {stats.get('status', 'unknown')}")
            except Exception as e:
                print(f"Could not retrieve stats: {e}")
        else:
            print("Statistics not available for current RAG agent")
        
        print("=" * 40)
    
    def _display_config(self):
        """Display current configuration"""
        print("\n⚙️ CURRENT CONFIGURATION")
        print("=" * 40)
        for key, value in self.config.items():
            if 'key' in key.lower() and value:
                # Hide API keys
                print(f"{key}: {'*' * 8}{value[-4:] if len(str(value)) > 4 else '****'}")
            else:
                print(f"{key}: {value}")
        print("=" * 40)
    
    def _display_help(self):
        """Display help information"""
        help_text = """
        📖 TRADING ASSISTANT HELP
        
        I can help you with:
        • Trading signals (buy/sell recommendations)
        • Market analysis and outlook
        • Risk assessment and portfolio management
        • Technical and fundamental analysis
        • Market news and economic data
        
        Examples:
        • "Should I buy Tesla stock?"
        • "What's the market risk today?"
        • "Analyze Bitcoin price"
        • "Give me a market summary"
        • "Is now a good time to invest in tech?"
        
        I cannot help with:
        • Retirement planning (401k, IRA, etc.)
        • Personal financial advice
        • Tax planning
        • Non-financial topics (weather, sports, etc.)
        
        Commands:
        • 'rag test' - Test document retrieval
        • 'stats' - Show system statistics
        • 'config' - Show configuration
        • 'clear' - Clear conversation
        • 'help' - Show this help
        • 'quit' - End session
        """
        print(help_text)
    
    async def run(self, mode: str = "interactive"):
        """
        Main entry point to run the system.
        
        Args:
            mode: "interactive" or "example"
        """
        try:
            # Initialize the assistant
            self.assistant = self._create_assistant()
            
            # Display system info
            self._display_banner()
            
            # Run in specified mode
            if mode.lower() == "example":
                await self.run_example_queries()
            else:
                await self.run_interactive_mode()
                
        except KeyboardInterrupt:
            print("\n\n🤖 System stopped by user.")
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            import traceback
            traceback.print_exc()
            return 1
        
        return 0


def main():
    """Command-line entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Trading Assistant System with Qdrant RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Interactive mode (default)
  %(prog)s --example          # Run example queries
  %(prog)s --config myconfig.json  # Load custom configuration
  %(prog)s --no-qdrant        # Run without Qdrant
  %(prog)s --local            # Use local Qdrant
  %(prog)s --cloud            # Use Qdrant Cloud
        
Environment Variables:
  USE_QDRANT=true/false       # Enable/disable Qdrant
  QDRANT_LOCAL=true/false     # Local vs Cloud mode
  QDRANT_URL=your-url         # Qdrant Cloud URL
  QDRANT_API_KEY=your-key     # Qdrant API key
        """
    )
    
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run example queries instead of interactive mode"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )
    
    parser.add_argument(
        "--no-qdrant",
        action="store_true",
        help="Disable Qdrant RAG (use simple agent)"
    )
    
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local Qdrant (default)"
    )
    
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use Qdrant Cloud (requires QDRANT_URL and QDRANT_API_KEY)"
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="trading_knowledge",
        help="Qdrant collection name"
    )
    
    args = parser.parse_args()
    
    # Set environment variables from command line args
    if args.no_qdrant:
        os.environ["USE_QDRANT"] = "false"
    
    if args.cloud:
        os.environ["QDRANT_LOCAL"] = "false"
        if not os.getenv("QDRANT_URL"):
            print("❌ Error: --cloud requires QDRANT_URL environment variable")
            return 1
    
    # Create and run the system
    system = TradingAssistantSystem(config_path=args.config)
    
    # Override config with command-line args
    if args.collection:
        system.config["qdrant_collection"] = args.collection
    
    # Determine mode
    mode = "example" if args.example else "interactive"
    
    # Run the system
    return asyncio.run(system.run(mode))


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)