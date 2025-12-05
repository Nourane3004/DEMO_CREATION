
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Statistics
from scipy import stats
from scipy.stats import norm

# Date handling
from datetime import datetime, timedelta
# Step 3: Define the Risk Agent Class
class RiskManagerAgent:
    """
    AI Agent for Risk Management that integrates inputs from:
    1. Macro Agent (economic trends)
    2. Quant/Stat Agent (price forecasts, volatility)
    3. Pattern Agent (technical signals, anomalies)
    Compatible with MultiAgentOrchestrator in the file "Orchestrator.py" .
    """

    def __init__(self, initial_capital=1000000, risk_tolerance="moderate"):
        """
        Initialize the Risk Manager Agent

        Parameters:
        -----------
        initial_capital : float
            Initial portfolio value
        risk_tolerance : str
            Risk appetite: "conservative", "moderate", or "aggressive"
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_tolerance = risk_tolerance

        # Risk parameters based on tolerance
        self.risk_params = self._set_risk_parameters(risk_tolerance)

        # Portfolio state
        self.portfolio = {
            'positions': {},  # {symbol: {'quantity': qty, 'avg_price': price}}
            'cash': initial_capital,
            'value': initial_capital,
            'history': []
        }

        # Models for risk assessment
        self.var_model = None
        self.es_model = None
        self.correlation_model = None

        # Store inputs from other agents
        self.macro_inputs = {}
        self.quant_inputs = {}
        self.pattern_inputs = {}

        # Risk metrics
        self.risk_metrics = {}

        print(f"✅ Risk Manager Agent initialized with {risk_tolerance} risk tolerance")
        print(f"📊 Initial Capital: ${initial_capital:,.2f}")

    def _set_risk_parameters(self, tolerance):
        """Set risk parameters based on tolerance level"""
        params = {
            "conservative": {
                "max_position_size": 0.05,  # 5% max per position
                "max_portfolio_risk": 0.10,  # 10% max portfolio VaR
                "stop_loss": 0.08,  # 8% stop loss
                "target_var_confidence": 0.95,
                "leverage_limit": 1.0,
                "diversification_min": 10  # Minimum number of positions
            },
            "moderate": {
                "max_position_size": 0.10,  # 10% max per position
                "max_portfolio_risk": 0.15,  # 15% max portfolio VaR
                "stop_loss": 0.12,  # 12% stop loss
                "target_var_confidence": 0.90,
                "leverage_limit": 1.5,
                "diversification_min": 5
            },
            "aggressive": {
                "max_position_size": 0.20,  # 20% max per position
                "max_portfolio_risk": 0.25,  # 25% max portfolio VaR
                "stop_loss": 0.15,  # 15% stop loss
                "target_var_confidence": 0.85,
                "leverage_limit": 2.0,
                "diversification_min": 3
            }
        }
        return params[tolerance]

    def receive_macro_inputs(self, macro_data):
        """
        Receive inputs from Macro Agent

        Parameters:
        -----------
        macro_data : dict
            Dictionary containing macroeconomic indicators
            Example: {'growth_trend': 'slowing', 'inflation': 'high',
                     'interest_rate_outlook': 'rising', 'sector_recommendations': ['energy']}
        """
        self.macro_inputs = macro_data
        print(f"📈 Received Macro Inputs: {len(macro_data)} indicators")

    def receive_quant_inputs(self, quant_data):
        """
        Receive inputs from Quant/Stat Agent

        Parameters:
        -----------
        quant_data : dict
            Dictionary containing quantitative forecasts
            Example: {
                'price_forecasts': {'SPY': -0.03, 'XLE': 0.05},
                'volatility_forecasts': {'SPY': 0.18, 'XLE': 0.22},
                'confidence_metrics': {'SPY': 0.85, 'XLE': 0.78},
                'regime_detection': 'high_volatility'
            }
        """
        self.quant_inputs = quant_data
        print(f"🔢 Received Quant Inputs: {len(quant_data)} forecasts")

    def receive_pattern_inputs(self, pattern_data):
        """
        Receive inputs from Pattern Agent

        Parameters:
        -----------
        pattern_data : dict
            Dictionary containing pattern-based signals
            Example: {
                'signals': {'XLE': 'bullish_breakout', 'TLT': 'bearish_momentum'},
                'anomalies': ['unusual_volume_SPY'],
                'technical_indicators': {'RSI': 'overbought', 'MACD': 'bullish'}
            }
        """
        self.pattern_inputs = pattern_data
        print(f"📊 Received Pattern Inputs: {len(pattern_data)} signals")

    def calculate_var(self, positions, price_data, confidence_level=0.95, method='historical'):
        """
        Calculate Value at Risk (VaR) for portfolio

        Parameters:
        -----------
        positions : dict
            Current portfolio positions
        price_data : DataFrame
            Historical price data
        confidence_level : float
            Confidence level for VaR (e.g., 0.95 for 95%)
        method : str
            'historical', 'parametric', or 'monte_carlo'

        Returns:
        --------
        var : float
            Value at Risk in absolute terms
        """
        if len(positions) == 0:
            return 0

        # Calculate portfolio returns
        portfolio_returns = self._calculate_portfolio_returns(positions, price_data)

        if method == 'historical':
            # Historical simulation method
            var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
            var_absolute = abs(var * self.current_capital)

        elif method == 'parametric':
            # Parametric (normal distribution) method
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)
            z_score = norm.ppf(1 - confidence_level)
            var = mean_return + z_score * std_return
            var_absolute = abs(var * self.current_capital)

        elif method == 'monte_carlo':
            # Monte Carlo simulation
            n_simulations = 10000
            mean_return = np.mean(portfolio_returns)
            std_return = np.std(portfolio_returns)

            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            var = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            var_absolute = abs(var * self.current_capital)

        return var_absolute

    def _calculate_portfolio_returns(self, positions, price_data):
        """Calculate historical portfolio returns"""
        # This is a simplified version - in practice, you'd use actual position weights
        portfolio_returns = []

        # For simplicity, use equal weighting if multiple positions
        for symbol, position in positions.items():
            if symbol in price_data.columns:
                returns = price_data[symbol].pct_change().dropna()
                portfolio_returns.append(returns.values)

        if portfolio_returns:
            # Average returns across positions
            portfolio_returns = np.mean(portfolio_returns, axis=0)
        else:
            portfolio_returns = np.array([0])

        return portfolio_returns

    def calculate_expected_shortfall(self, positions, price_data, confidence_level=0.95):
        """
        Calculate Expected Shortfall (CVaR)

        Parameters:
        -----------
        positions : dict
            Current portfolio positions
        price_data : DataFrame
            Historical price data

        Returns:
        --------
        es : float
            Expected Shortfall in absolute terms
        """
        portfolio_returns = self._calculate_portfolio_returns(positions, price_data)

        # Calculate VaR first
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)

        # Calculate ES as average of returns worse than VaR
        es_returns = portfolio_returns[portfolio_returns <= var]
        es = np.mean(es_returns) if len(es_returns) > 0 else var
        es_absolute = abs(es * self.current_capital)

        return es_absolute

    def calculate_correlation_matrix(self, price_data):
        """
        Calculate correlation matrix for assets

        Parameters:
        -----------
        price_data : DataFrame
            Historical price data

        Returns:
        --------
        correlation_matrix : DataFrame
            Correlation matrix
        """
        returns = price_data.pct_change().dropna()
        correlation_matrix = returns.corr()
        return correlation_matrix

    def stress_test_scenarios(self, positions, price_data, scenarios):
        """
        Perform stress testing under different scenarios

        Parameters:
        -----------
        positions : dict
            Current portfolio positions
        price_data : DataFrame
            Historical price data
        scenarios : list of dict
            List of scenario definitions

        Returns:
        --------
        scenario_results : dict
            Results for each scenario
        """
        scenario_results = {}

        for scenario in scenarios:
            scenario_name = scenario['name']
            shock_factors = scenario['shocks']  # e.g., {'SPY': -0.20, 'TLT': 0.10}

            # Calculate portfolio impact
            portfolio_impact = 0
            for symbol, position in positions.items():
                if symbol in shock_factors:
                    position_value = position['quantity'] * price_data[symbol].iloc[-1]
                    shock = shock_factors[symbol]
                    portfolio_impact += position_value * shock

            scenario_results[scenario_name] = {
                'portfolio_impact': portfolio_impact,
                'percent_impact': portfolio_impact / self.current_capital * 100
            }

        return scenario_results

    def generate_hedging_recommendations(self, positions, price_data):
        """
        Generate hedging recommendations based on portfolio risk

        Parameters:
        -----------
        positions : dict
            Current portfolio positions
        price_data : DataFrame
            Historical price data

        Returns:
        --------
        recommendations : dict
            Hedging recommendations
        """
        recommendations = {
            'hedge_instruments': [],
            'hedge_ratios': {},
            'recommended_actions': []
        }

        # Calculate portfolio beta (simplified)
        if 'SPY' in price_data.columns and len(positions) > 0:
            portfolio_returns = self._calculate_portfolio_returns(positions, price_data)
            spy_returns = price_data['SPY'].pct_change().dropna().values

            # Align lengths
            min_len = min(len(portfolio_returns), len(spy_returns))
            beta = np.cov(portfolio_returns[:min_len], spy_returns[:min_len])[0, 1] / np.var(spy_returns[:min_len])

            # Suggest hedging based on beta
            if beta > 1.2:
                recommendations['hedge_instruments'].append('Put options on SPY')
                recommendations['hedge_ratios']['SPY_puts'] = 0.1  # Hedge 10% of exposure
                recommendations['recommended_actions'].append('Consider buying protective puts')
            elif beta < -0.5:
                recommendations['hedge_instruments'].append('Call options on SPY')
                recommendations['hedge_ratios']['SPY_calls'] = 0.05
                recommendations['recommended_actions'].append('Consider call options for short exposure')

        # Add recommendations from macro inputs
        if 'sector_recommendations' in self.macro_inputs:
            for sector in self.macro_inputs['sector_recommendations']:
                recommendations['recommended_actions'].append(f"Consider hedging {sector} sector exposure")

        return recommendations

    def portfolio_allocation_recommendation(self):
        """
        Generate portfolio allocation recommendations based on all inputs
        """
        recommendations = {
            'asset_allocation': {},
            'position_sizing': {},
            'risk_adjustments': []
        }

        # Integrate inputs from all agents
        risk_adjustment = 1.0

        # Adjust based on macro conditions
        if 'growth_trend' in self.macro_inputs:
            if self.macro_inputs['growth_trend'] == 'slowing':
                risk_adjustment *= 0.7  # Reduce exposure
                recommendations['risk_adjustments'].append('Reduce equity exposure due to slowing growth')

        # Adjust based on volatility forecasts
        if 'volatility_forecasts' in self.quant_inputs:
            avg_vol = np.mean(list(self.quant_inputs['volatility_forecasts'].values()))
            if avg_vol > 0.25:  # High volatility
                risk_adjustment *= 0.8
                recommendations['risk_adjustments'].append('Reduce position sizes due to high volatility')

        # Adjust based on pattern signals
        if 'signals' in self.pattern_inputs:
            bullish_signals = sum(1 for s in self.pattern_inputs['signals'].values() if 'bullish' in s)
            bearish_signals = sum(1 for s in self.pattern_inputs['signals'].values() if 'bearish' in s)

            if bullish_signals > bearish_signals * 2:
                risk_adjustment *= 1.1  # Slightly increase exposure
            elif bearish_signals > bullish_signals * 2:
                risk_adjustment *= 0.9  # Slightly decrease exposure

        # Apply risk tolerance
        max_position = self.risk_params['max_position_size'] * risk_adjustment

        recommendations['position_sizing']['max_position_percent'] = max_position * 100
        recommendations['position_sizing']['recommended_diversification'] = self.risk_params['diversification_min']

        # Suggested allocation based on inputs
        recommendations['asset_allocation'] = {
            'equity': 60 * risk_adjustment,
            'bonds': 30,
            'cash': 10,
            'alternatives': 0
        }

        return recommendations

    def generate_risk_alerts(self):
        """
        Generate risk alerts based on integrated analysis
        """
        alerts = []

        # Check portfolio concentration
        if len(self.portfolio['positions']) < self.risk_params['diversification_min']:
            alerts.append({
                'level': 'warning',
                'message': f'Portfolio under-diversified: {len(self.portfolio["positions"])} positions',
                'action': f'Consider adding positions to reach minimum {self.risk_params["diversification_min"]}'
            })

        # Check individual position sizes
        for symbol, position in self.portfolio['positions'].items():
            position_value = position['quantity'] * position.get('current_price', position['avg_price'])
            position_percent = position_value / self.current_capital

            if position_percent > self.risk_params['max_position_size']:
                alerts.append({
                    'level': 'critical',
                    'message': f'Position {symbol} too large: {position_percent:.1%} of portfolio',
                    'action': f'Reduce position to under {self.risk_params["max_position_size"]*100:.1f}%'
                })

        # Check for anomalies from pattern agent
        if 'anomalies' in self.pattern_inputs and self.pattern_inputs['anomalies']:
            alerts.append({
                'level': 'warning',
                'message': f'Pattern Agent detected {len(self.pattern_inputs["anomalies"])} anomalies',
                'action': 'Review pattern signals and consider adjusting positions'
            })

        return alerts

    def integrate_and_decide(self, price_data):
        """
        Main method: Integrate all inputs and generate final decisions

        Parameters:
        -----------
        price_data : DataFrame
            Current and historical price data

        Returns:
        --------
        decisions : dict
            Comprehensive risk management decisions
        """
        print("\n" + "="*60)
        print("RISK AGENT: INTEGRATING ALL INPUTS AND GENERATING DECISIONS")
        print("="*60)

        # Calculate risk metrics
        var = self.calculate_var(self.portfolio['positions'], price_data,
                                 self.risk_params['target_var_confidence'])
        es = self.calculate_expected_shortfall(self.portfolio['positions'], price_data,
                                              self.risk_params['target_var_confidence'])
        correlation = self.calculate_correlation_matrix(price_data)

        # Store metrics
        self.risk_metrics = {
            'value_at_risk': var,
            'var_percent': var / self.current_capital * 100,
            'expected_shortfall': es,
            'es_percent': es / self.current_capital * 100,
            'correlation_matrix': correlation
        }

        # Generate stress test scenarios
        scenarios = [
            {'name': 'Market Crash', 'shocks': {'SPY': -0.30, 'TLT': 0.10}},
            {'name': 'Rate Shock', 'shocks': {'SPY': -0.15, 'TLT': -0.20}},
            {'name': 'Sector Rotation', 'shocks': {'XLE': 0.25, 'XLK': -0.15}}
        ]
        stress_results = self.stress_test_scenarios(self.portfolio['positions'], price_data, scenarios)

        # Generate recommendations
        hedging_recs = self.generate_hedging_recommendations(self.portfolio['positions'], price_data)
        allocation_recs = self.portfolio_allocation_recommendation()
        risk_alerts = self.generate_risk_alerts()

        # Compile final decisions
        decisions = {
            'risk_assessment': {
                'portfolio_value': self.current_capital,
                'value_at_risk': var,
                'var_percent': var / self.current_capital * 100,
                'expected_shortfall': es,
                'max_drawdown_limit': self.current_capital * self.risk_params['stop_loss'],
                'risk_tolerance': self.risk_tolerance
            },
            'recommendations': {
                'hedging': hedging_recs,
                'allocation': allocation_recs,
                'position_sizing': {
                    'max_per_position': self.risk_params['max_position_size'] * 100,
                    'current_positions': len(self.portfolio['positions'])
                }
            },
            'stress_test_results': stress_results,
            'risk_alerts': risk_alerts,
            'integrated_signals': {
                'macro_context': self.macro_inputs.get('growth_trend', 'neutral'),
                'quant_confidence': np.mean(list(self.quant_inputs.get('confidence_metrics', {}).values()))
                                  if self.quant_inputs.get('confidence_metrics') else 0.5,
                'pattern_signals': len(self.pattern_inputs.get('signals', {}))
            },
            'final_decisions': self._generate_final_decisions()
        }

        return decisions

    def _generate_final_decisions(self):
        """Generate final trading/hedging decisions"""
        decisions = []

        # Example logic based on integrated analysis
        if self.macro_inputs.get('growth_trend') == 'slowing':
            decisions.append("Reduce overall market exposure by 20%")

        if 'volatility_forecasts' in self.quant_inputs:
            if np.mean(list(self.quant_inputs['volatility_forecasts'].values())) > 0.25:
                decisions.append("Increase cash position and consider volatility hedging")

        if self.pattern_inputs.get('signals'):
            bullish_count = sum(1 for s in self.pattern_inputs['signals'].values() if 'bullish' in s)
            if bullish_count > 3:
                decisions.append("Selectively add to positions with strongest bullish patterns")

        if not decisions:
            decisions.append("Maintain current portfolio with regular monitoring")

        return decisions

    def visualize_risk_metrics(self, decisions):
        """
        Visualize risk metrics and decisions

        Parameters:
        -----------
        decisions : dict
            Output from integrate_and_decide method
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Value at Risk', 'Stress Test Results',
                          'Portfolio Allocation', 'Risk Alerts'),
            specs=[[{'type': 'indicator'}, {'type': 'bar'}],
                   [{'type': 'pie'}, {'type': 'table'}]]
        )

        # VaR Gauge
        var_percent = decisions['risk_assessment']['var_percent']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=var_percent,
                title={'text': f"VaR ({self.risk_params['target_var_confidence']*100:.0f}% Confidence)"},
                gauge={'axis': {'range': [0, 30]},
                      'steps': [
                          {'range': [0, 10], 'color': "lightgreen"},
                          {'range': [10, 20], 'color': "yellow"},
                          {'range': [20, 30], 'color': "red"}],
                      'threshold': {'line': {'color': "black", 'width': 4},
                                   'thickness': 0.75,
                                   'value': self.risk_params['max_portfolio_risk']*100}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )

        # Stress Test Results
        scenarios = list(decisions['stress_test_results'].keys())
        impacts = [decisions['stress_test_results'][s]['percent_impact'] for s in scenarios]

        colors = ['red' if impact < -10 else 'orange' if impact < -5 else 'yellow'
                 for impact in impacts]

        fig.add_trace(
            go.Bar(x=scenarios, y=impacts, marker_color=colors,
                  name='Portfolio Impact %'),
            row=1, col=2
        )

        # Portfolio Allocation
        allocation = decisions['recommendations']['allocation']['asset_allocation']
        labels = list(allocation.keys())
        values = list(allocation.values())

        fig.add_trace(
            go.Pie(labels=labels, values=values, hole=0.3,
                  name='Recommended Allocation'),
            row=2, col=1
        )

        # Risk Alerts Table
        if decisions['risk_alerts']:
            alert_levels = [alert['level'] for alert in decisions['risk_alerts']]
            alert_messages = [alert['message'] for alert in decisions['risk_alerts']]
            alert_actions = [alert['action'] for alert in decisions['risk_alerts']]

            fig.add_trace(
                go.Table(
                    header=dict(values=['Level', 'Message', 'Recommended Action'],
                               fill_color='lightgrey'),
                    cells=dict(values=[alert_levels, alert_messages, alert_actions],
                              fill_color=[['lightcoral' if level=='critical' else
                                          'lightyellow' if level=='warning' else 'lightblue'
                                          for level in alert_levels]] * 3),
                ),
                row=2, col=2
            )

        fig.update_layout(height=800, width=1000,
                         title_text="Risk Manager Agent Dashboard",
                         showlegend=False)
        fig.show()

        # Step 4: Demo and Testing

# Create sample price data for testing
def create_sample_price_data():
    """Create sample price data for demonstration"""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='B')
    n_days = len(dates)

    # Generate synthetic price data
    np.random.seed(42)

    # SPY (S&P 500 ETF)
    spy_returns = np.random.normal(0.0005, 0.015, n_days)
    spy_prices = 400 * np.exp(np.cumsum(spy_returns))

    # XLE (Energy ETF)
    xle_returns = np.random.normal(0.0003, 0.018, n_days)
    xle_prices = 80 * np.exp(np.cumsum(xle_returns))

    # TLT (Bond ETF)
    tlt_returns = np.random.normal(0.0001, 0.008, n_days)
    tlt_prices = 100 * np.exp(np.cumsum(tlt_returns))

    price_data = pd.DataFrame({
        'SPY': spy_prices,
        'XLE': xle_prices,
        'TLT': tlt_prices
    }, index=dates)

    return price_data

# Initialize the Risk Manager Agent
risk_agent = RiskManagerAgent(initial_capital=1000000, risk_tolerance="moderate")

# Simulate receiving inputs from other agents
print("\n📥 SIMULATING INPUTS FROM OTHER AGENTS")
print("-" * 40)

# From Macro Agent
macro_data = {
    'growth_trend': 'slowing',
    'inflation': 'high',
    'interest_rate_outlook': 'rising',
    'sector_recommendations': ['energy', 'defensive'],
    'economic_regime': 'late_cycle'
}
risk_agent.receive_macro_inputs(macro_data)

# From Quant/Stat Agent
quant_data = {
    'price_forecasts': {'SPY': -0.03, 'XLE': 0.05, 'TLT': 0.02},
    'volatility_forecasts': {'SPY': 0.18, 'XLE': 0.22, 'TLT': 0.12},
    'confidence_metrics': {'SPY': 0.85, 'XLE': 0.78, 'TLT': 0.91},
    'regime_detection': 'high_volatility',
    'market_regime': 'bearish'
}
risk_agent.receive_quant_inputs(quant_data)

# From Pattern Agent
pattern_data = {
    'signals': {
        'XLE': 'bullish_breakout',
        'TLT': 'bearish_momentum',
        'SPY': 'sideways_consolidation'
    },
    'anomalies': ['unusual_volume_SPY', 'gap_down_XLE'],
    'technical_indicators': {
        'RSI_SPY': 'neutral',
        'MACD_XLE': 'bullish_crossover',
        'BB_TLT': 'oversold'
    }
}
risk_agent.receive_pattern_inputs(pattern_data)

# Simulate some portfolio positions
risk_agent.portfolio['positions'] = {
    'SPY': {'quantity': 100, 'avg_price': 420, 'current_price': 415},
    'XLE': {'quantity': 500, 'avg_price': 78, 'current_price': 82},
    'TLT': {'quantity': 200, 'avg_price': 98, 'current_price': 101}
}

# Update portfolio value
risk_agent.current_capital = 1050000  # Simulated growth

# Get price data
price_data = create_sample_price_data()

# Step 5: Run the Risk Agent
decisions = risk_agent.integrate_and_decide(price_data)

# Display key decisions
print("\n🔑 KEY RISK DECISIONS:")
print("-" * 40)
for i, decision in enumerate(decisions['final_decisions'], 1):
    print(f"{i}. {decision}")

print("\n⚠️  RISK ALERTS:")
print("-" * 40)
if decisions['risk_alerts']:
    for alert in decisions['risk_alerts']:
        print(f"[{alert['level'].upper()}] {alert['message']}")
        print(f"   → Action: {alert['action']}")
else:
    print("No critical risk alerts at this time.")

print("\n📊 RISK METRICS:")
print("-" * 40)
print(f"Portfolio Value: ${decisions['risk_assessment']['portfolio_value']:,.2f}")
print(f"Value at Risk (95%): ${decisions['risk_assessment']['value_at_risk']:,.2f} ({decisions['risk_assessment']['var_percent']:.2f}%)")
print(f"Expected Shortfall: ${decisions['risk_assessment']['expected_shortfall']:,.2f} ({decisions['risk_assessment']['expected_shortfall'] / decisions['risk_assessment']['portfolio_value'] * 100:.2f}%)")
print(f"Max Drawdown Limit: ${decisions['risk_assessment']['max_drawdown_limit']:,.2f}")

# Step 6: Visualize Results
print("\n📈 GENERATING RISK DASHBOARD...")
risk_agent.visualize_risk_metrics(decisions)


# Step 7: Advanced Features - ML-based Risk Prediction

class MLRiskPredictor:
    """
    Machine Learning component for advanced risk prediction
    """

    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}

    def prepare_features(self, macro_data, quant_data, pattern_data, market_data):
        """
        Prepare features from all agent inputs and market data
        """
        features = {}

        # Macro features
        macro_mapping = {'slowing': -1, 'neutral': 0, 'accelerating': 1}
        features['growth_trend'] = macro_mapping.get(macro_data.get('growth_trend', 'neutral'), 0)
        features['inflation_high'] = 1 if macro_data.get('inflation') == 'high' else 0

        # Quant features
        if 'volatility_forecasts' in quant_data:
            features['avg_volatility'] = np.mean(list(quant_data['volatility_forecasts'].values()))
        if 'confidence_metrics' in quant_data:
            features['avg_confidence'] = np.mean(list(quant_data['confidence_metrics'].values()))

        # Pattern features
        if 'signals' in pattern_data:
            bullish_count = sum(1 for s in pattern_data['signals'].values() if 'bullish' in s)
            bearish_count = sum(1 for s in pattern_data['signals'].values() if 'bearish' in s)
            features['signal_ratio'] = bullish_count / (bearish_count + 1)

        # Market features
        features['market_returns'] = market_data['returns'].iloc[-1] if 'returns' in market_data.columns else 0
        features['market_volatility'] = market_data['volatility'].iloc[-1] if 'volatility' in market_data.columns else 0

        return pd.DataFrame([features])

    def train_var_model(self, X_train, y_train, model_type='xgboost'):
        """
        Train ML model for VaR prediction
        """
        if model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        self.models['var_model'] = model

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance['var_model'] = model.feature_importances_

        return model

    def predict_risk(self, features):
        """
        Predict risk metrics using trained models
        """
        predictions = {}

        if 'var_model' in self.models:
            # Scale features
            features_scaled = self.scaler.transform(features) if hasattr(self.scaler, 'scale_') else features

            # Predict VaR
            var_pred = self.models['var_model'].predict(features_scaled)
            predictions['var'] = var_pred[0]

        return predictions

# Example usage of ML component
print("\n🤖 INITIALIZING ML RISK PREDICTOR...")
print("-" * 40)

ml_predictor = MLRiskPredictor()

# Prepare sample training data (in practice, you'd use historical data)
# This is just a demonstration structure
print("ML Risk Predictor initialized successfully!")
print("To train models, you would need:")
print("1. Historical feature data from all agents")
print("2. Historical actual VaR/risk metrics")
print("3. Proper train/test split and validation")


# Step 8: Export/Import Agent State

import json
import pickle

class RiskAgentPersistence:
    """
    Handle saving and loading of Risk Agent state
    """

    @staticmethod
    def save_agent(agent, filepath='risk_agent_state.pkl'):
        """
        Save agent state to file
        """
        state = {
            'portfolio': agent.portfolio,
            'risk_params': agent.risk_params,
            'current_capital': agent.current_capital,
            'risk_tolerance': agent.risk_tolerance,
            'risk_metrics': agent.risk_metrics
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        print(f"✅ Agent state saved to {filepath}")

    @staticmethod
    def load_agent(filepath='risk_agent_state.pkl'):
        """
        Load agent state from file
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Create new agent with saved state
        agent = RiskManagerAgent(
            initial_capital=state['current_capital'],
            risk_tolerance=state['risk_tolerance']
        )

        agent.portfolio = state['portfolio']
        agent.risk_params = state['risk_params']
        agent.risk_metrics = state['risk_metrics']

        print(f"✅ Agent state loaded from {filepath}")
        return agent

# Example: Save current agent state
persistence = RiskAgentPersistence()
persistence.save_agent(risk_agent, 'risk_agent_demo.pkl')

print("\n" + "="*60)
print("RISK AGENT IMPLEMENTATION COMPLETE!")
print("="*60)
print("\nSummary of implemented features:")
print("1. ✅ Risk Manager Agent class with configurable risk tolerance")
print("2. ✅ Methods to receive inputs from Macro, Quant, and Pattern agents")
print("3. ✅ VaR and Expected Shortfall calculations (multiple methods)")
print("4. ✅ Stress testing with customizable scenarios")
print("5. ✅ Hedging recommendations engine")
print("6. ✅ Portfolio allocation suggestions")
print("7. ✅ Risk alert generation system")
print("8. ✅ Integrated decision-making combining all inputs")
print("9. ✅ Visualization dashboard with Plotly")
print("10. ✅ ML Risk Predictor class (ready for training)")
print("11. ✅ Persistence layer for saving/loading agent state")
print("\nThe agent is now ready to integrate with your other AI agents!")


risk_agent = RiskManagerAgent(initial_capital=1000000, risk_tolerance="moderate")

decisions = risk_agent.integrate_and_decide(price_data)


# Risk metrics
print(decisions['risk_assessment'])

# Recommendations
print(decisions['recommendations'])

# Alerts
for alert in decisions['risk_alerts']:
    print(f"Alert: {alert['message']}")

# Final decisions
for decision in decisions['final_decisions']:
    print(f"Decision: {decision}")

risk_agent.visualize_risk_metrics(decisions)