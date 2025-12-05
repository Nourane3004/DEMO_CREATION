# macro_vision_independent.py - Agent d'Analyse Macroéconomique Indépendant

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import json
import requests
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import concurrent.futures
warnings.filterwarnings('ignore')

# Pour VAR/VECM
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

class MacroVisionAgent:
    """
    MACRO-VISION: Agent d'analyse macroéconomique et géopolitique,
    Compatible with MultiAgentOrchestrator in the file "Orchestrator.py" .
    """

    def __init__(self):
        self.macro_data = {}
        self.user_profile = {}
        self.portfolio_exposure = {}  # Exposition sectorielle/géographique
        self.current_market_regime = "unknown"

        # Cache pour les données
        self.data_cache = {}
        self.last_update = None

        # Paramètres par défaut
        self.default_params = {
            'country_focus': 'US',
            'analysis_horizon': 'medium',
            'risk_tolerance': 'moderate',
            'investment_universe': ['equities', 'bonds', 'commodities']
        }

    

    def parse_portfolio_exposure(self, input_str: str):
        """Parse l'exposition du portefeuille"""
        exposures = {}
        try:
            for item in input_str.split(','):
                if ':' in item:
                    sector, pct = item.split(':')
                    sector = sector.strip()
                    pct = float(pct.strip())
                    if 0 <= pct <= 100:
                        exposures[sector] = pct
        except:
            print("⚠️  Format d'exposition invalide, ignoré")

        self.portfolio_exposure = exposures
        if exposures:
            print(f"   Exposition chargée: {len(exposures)} secteurs")


    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main method called by orchestrator.
        
        Args:
            context: Dictionary from orchestrator containing:
                - query: User's question
                - market_data: Current market prices/vol
                - rag_docs: Documents from RAG agent (if use_rag=True)
                - rag_summary: LLM summary of RAG docs (if available)
                - price_series: Historical price data
                - risk_params: Risk parameters
                - timestamp: Request timestamp
        
        Returns:
            Dictionary with macro analysis results
        """
        # Extract data from context (with defaults)
        query = context.get("query", "")
        market_data = context.get("market_data", {})
        rag_docs = context.get("rag_docs", [])
        rag_summary = context.get("rag_summary", "")
        price_series = context.get("price_series", [])
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Analyze RAG documents if available
        economic_indicators = self._extract_economic_indicators(rag_docs)
        
        # Generate macro summary based on available data
        macro_summary = self._generate_macro_summary(
            rag_docs=rag_docs,
            rag_summary=rag_summary,
            market_data=market_data,
            query=query
        )
        
        # Determine market regime
        regime = self._determine_regime(
            market_data=market_data,
            economic_indicators=economic_indicators,
            price_series=price_series
        )
        
        # Return results in standard format
        return {
            "agent": "MACRO-INSIGHT",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "macro_summary": macro_summary,
            "regime": regime,
            "economic_indicators": economic_indicators,
            "rag_docs_used": len(rag_docs),
            "market_data_keys": list(market_data.keys()),
            "query_analyzed": query[:100] if query else ""
        }
    

    def fetch_economic_data(self, force_refresh: bool = False):
        """Récupère les données économiques de différentes sources"""
        print("\n📊 COLLECTE DES DONNÉES MACROÉCONOMIQUES")
        print("-" * 40)

        # Vérifier le cache
        if not force_refresh and self.last_update and (datetime.now() - self.last_update).hours < 1:
            print("Utilisation des données en cache...")
            return self.macro_data

        country = self.user_profile.get('country_focus', 'US')

        try:
            # 1. Indicateurs économiques principaux
            print("1. Indicateurs économiques...")
            economic_indicators = self.get_economic_indicators(country)

            # 2. Données de marché
            print("2. Données de marché...")
            market_data = self.get_market_data()

            # 3. Politiques monétaires
            print("3. Politiques monétaires...")
            monetary_policy = self.get_monetary_policy_data(country)

            # 4. Données géopolitiques
            print("4. Données géopolitiques...")
            geopolitical_data = self.get_geopolitical_data()

            # 5. Sentiment et flux
            print("5. Sentiment de marché...")
            sentiment_data = self.get_market_sentiment()

            # Fusionner toutes les données
            self.macro_data = {
                'economic_indicators': economic_indicators,
                'market_data': market_data,
                'monetary_policy': monetary_policy,
                'geopolitical': geopolitical_data,
                'sentiment': sentiment_data,
                'last_updated': datetime.now().isoformat()
            }

            self.last_update = datetime.now()

            print(f"✅ Données collectées: {len(economic_indicators)} indicateurs économiques")
            print(f"   Données marché: {len(market_data)} séries")
            print(f"   Données géopolitiques: {len(geopolitical_data)} indicateurs")

        except Exception as e:
            print(f"⚠️  Erreur collecte données: {e}")
            print("Utilisation de données simulées...")
            self.load_sample_data()

        return self.macro_data

    def get_economic_indicators(self, country: str) -> Dict:
        """Récupère les indicateurs économiques"""
        # En production, intégrer avec FRED, OECD, TradingEconomics APIs

        indicators = {}

        # Simulation de données selon le pays
        if country == 'US':
            indicators = {
                'GDP_GROWTH': {
                    'value': 2.7, 'unit': '% YoY', 'trend': 'stable',
                    'forecast': 2.5, 'volatility': 0.3, 'importance': 10
                },
                'CPI_INFLATION': {
                    'value': 3.1, 'unit': '% YoY', 'trend': 'decreasing',
                    'forecast': 2.8, 'volatility': 0.4, 'importance': 10
                },
                'UNEMPLOYMENT_RATE': {
                    'value': 3.7, 'unit': '%', 'trend': 'stable',
                    'forecast': 3.8, 'volatility': 0.2, 'importance': 8
                },
                'ISM_MANUFACTURING': {
                    'value': 49.5, 'unit': 'index', 'trend': 'increasing',
                    'forecast': 50.2, 'volatility': 1.5, 'importance': 7
                },
                'CONSUMER_CONFIDENCE': {
                    'value': 110.5, 'unit': 'index', 'trend': 'stable',
                    'forecast': 112.0, 'volatility': 3.0, 'importance': 6
                },
                'RETAIL_SALES': {
                    'value': 0.6, 'unit': '% MoM', 'trend': 'positive',
                    'forecast': 0.4, 'volatility': 0.5, 'importance': 7
                }
            }
        elif country == 'EU':
            # Données pour Europe
            indicators = {
                'GDP_GROWTH': {'value': 0.5, 'unit': '% QoQ', 'trend': 'weak'},
                'CPI_INFLATION': {'value': 2.9, 'unit': '% YoY', 'trend': 'decreasing'},
                'UNEMPLOYMENT_RATE': {'value': 6.5, 'unit': '%', 'trend': 'stable'}
            }
        else:
            # Données génériques
            indicators = {
                'GDP_GROWTH': {'value': 2.0, 'unit': '% YoY', 'trend': 'stable'},
                'CPI_INFLATION': {'value': 3.5, 'unit': '% YoY', 'trend': 'stable'},
                'UNEMPLOYMENT_RATE': {'value': 5.0, 'unit': '%', 'trend': 'stable'}
            }

        return indicators

    def get_market_data(self) -> Dict:
        """Récupère les données de marché"""
        import yfinance as yf

        market_data = {}
        tickers = {
            'SP500': '^GSPC',
            'NASDAQ': '^IXIC',
            'DOW': '^DJI',
            'VIX': '^VIX',
            'US10Y': '^TNX',
            'DXY': 'DX-Y.NYB',
            'GOLD': 'GC=F',
            'OIL': 'CL=F'
        }

        try:
            for name, ticker in tickers.items():
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period='1mo')

                    if not hist.empty:
                        current = hist['Close'].iloc[-1]
                        prev = hist['Close'].iloc[-2] if len(hist) > 1 else current
                        change_pct = ((current - prev) / prev * 100) if prev > 0 else 0

                        market_data[name] = {
                            'price': float(current),
                            'change_pct': float(change_pct),
                            'trend': 'up' if change_pct > 0 else 'down',
                            'volatility': float(hist['Close'].pct_change().std() * 100)
                        }
                except:
                    continue

            print(f"   {len(market_data)} instruments de marché récupérés")

        except Exception as e:
            print(f"   Erreur données marché: {e}")
            # Données simulées en fallback
            market_data = self.get_sample_market_data()

        return market_data

    def get_monetary_policy_data(self, country: str) -> Dict:
        """Récupère les données de politique monétaire"""

        if country == 'US':
            return {
                'FED_FUNDS_RATE': {'value': 5.25, 'trend': 'stable', 'next_meeting': '2024-03-20'},
                'FED_STANCE': {'value': 'hawkish', 'description': 'Data dependent, higher for longer'},
                'BALANCE_SHEET': {'value': 'QT ongoing', 'pace': '$95B/month'},
                'RATE_PROBABILITIES': {
                    'hold': 0.80,
                    'hike': 0.15,
                    'cut': 0.05
                }
            }
        elif country == 'EU':
            return {
                'ECB_RATE': {'value': 4.0, 'trend': 'stable'},
                'ECB_STANCE': {'value': 'neutral', 'description': 'Monitoring inflation'}
            }
        else:
            return {
                'POLICY_RATE': {'value': 4.5, 'trend': 'stable'},
                'STANCE': {'value': 'neutral'}
            }

    def get_geopolitical_data(self) -> Dict:
        """Récupère les données géopolitiques"""
        # En production, intégrer GDELT Project

        return {
            'GLOBAL_RISK_INDEX': {
                'value': 0.35,  # 0-1
                'trend': 'increasing',
                'level': 'moderate',
                'hotspots': ['Middle East', 'Ukraine', 'South China Sea']
            },
            'TRADE_TENSIONS': {
                'value': 0.25,
                'trend': 'stable',
                'description': 'US-China relations stable'
            },
            'ELECTION_RISK': {
                'value': 0.40,
                'trend': 'increasing',
                'upcoming_elections': ['US 2024', 'EU 2024']
            }
        }

    def get_market_sentiment(self) -> Dict:
        """Récupère les données de sentiment"""

        return {
            'AAII_SENTIMENT': {
                'bullish': 42.3,
                'neutral': 30.1,
                'bearish': 27.6,
                'bull_bear_spread': 14.7
            },
            'PUT_CALL_RATIO': {
                'value': 0.85,
                'interpretation': 'neutral',
                'trend': 'decreasing'
            },
            'FEAR_GREED_INDEX': {
                'value': 65,
                'level': 'greed',
                'trend': 'increasing'
            }
        }

    def get_sample_market_data(self) -> Dict:
        """Retourne des données de marché simulées"""
        return {
            'SP500': {'price': 5100.0, 'change_pct': 0.5, 'trend': 'up'},
            'NASDAQ': {'price': 16000.0, 'change_pct': 0.8, 'trend': 'up'},
            'VIX': {'price': 14.5, 'change_pct': -2.0, 'trend': 'down'},
            'US10Y': {'price': 4.25, 'change_pct': 0.1, 'trend': 'stable'},
            'GOLD': {'price': 2050.0, 'change_pct': 1.2, 'trend': 'up'}
        }

    def load_sample_data(self):
        """Charge des données d'exemple complètes"""
        print("Chargement données d'exemple...")

        self.macro_data = {
            'economic_indicators': self.get_economic_indicators('US'),
            'market_data': self.get_sample_market_data(),
            'monetary_policy': self.get_monetary_policy_data('US'),
            'geopolitical': self.get_geopolitical_data(),
            'sentiment': self.get_market_sentiment(),
            'last_updated': datetime.now().isoformat()
        }

    def run_var_analysis(self):
        """Exécute l'analyse VAR (Vector Autoregression)"""
        print("\n🔬 EXÉCUTION DE L'ANALYSE VAR")
        print("-" * 40)

        try:
            # Préparer les séries temporelles
            time_series = self.prepare_time_series_data()

            if time_series is None or len(time_series) < 24:
                print("⚠️  Données temporelles insuffisantes pour VAR")
                return None

            # Créer le modèle VAR
            model = VAR(time_series)

            # Sélection automatique du lag
            lag_results = model.select_order(maxlags=8)
            optimal_lags = lag_results.aic

            # Ajuster le modèle
            results = model.fit(maxlags=optimal_lags)

            # Générer des prévisions
            forecast_steps = self.get_forecast_horizon()
            forecast = results.forecast(time_series.values[-results.k_ar:], forecast_steps)

            # Analyser les impulsions-réponses
            irf = results.irf(periods=10)

            # Calculer la décomposition de la variance
            fevd = results.fevd(10)

            var_results = {
                'optimal_lags': int(optimal_lags),
                'forecast': self.format_forecast(forecast, time_series.columns, forecast_steps),
                'variance_decomposition': self.summarize_variance_decomposition(fevd),
                'granger_causality': self.calculate_granger_causality(results, time_series.columns),
                'model_metrics': {
                    'aic': float(results.aic),
                    'bic': float(results.bic),
                    'rsquared_avg': float(results.rsquared.mean())
                },
                'significant_variables': self.identify_significant_vars(results)
            }

            print(f"✅ Analyse VAR terminée (lag optimal: {optimal_lags})")
            return var_results

        except Exception as e:
            print(f"❌ Erreur analyse VAR: {e}")
            return None

    def prepare_time_series_data(self):
        """Prépare les séries temporelles pour VAR"""
        # En production, récupérer des données historiques
        # Pour l'exemple, générer des séries simulées

        np.random.seed(42)
        periods = 60  # 5 ans de données mensuelles

        dates = pd.date_range(end=datetime.now(), periods=periods, freq='M')

        data = {
            'GDP_GROWTH': np.random.normal(2.5, 0.8, periods).cumsum() / 10 + 2.0,
            'CPI_INFLATION': np.random.normal(3.0, 0.5, periods).cumsum() / 10 + 2.5,
            'UNEMPLOYMENT': np.random.normal(3.8, 0.3, periods),
            'SP500_RETURNS': np.random.normal(0.8, 2.0, periods).cumsum() / 10 + 8.0,
            'INTEREST_RATE': np.random.normal(4.0, 0.2, periods)
        }

        df = pd.DataFrame(data, index=dates)

        # Stationnariser les séries (différence première)
        df_stationary = df.diff().dropna()

        return df_stationary

    def get_forecast_horizon(self):
        """Détermine l'horizon de prévision basé sur le profil utilisateur"""
        horizon_map = {
            'court': 3,    # 3 mois
            'moyen': 6,    # 6 mois
            'long': 12     # 12 mois
        }
        return horizon_map.get(self.user_profile.get('analysis_horizon', 'moyen'), 6)

    def format_forecast(self, forecast, columns, steps):
        """Formate les prévisions VAR"""
        formatted = {}
        for i, col in enumerate(columns):
            formatted[col] = {
                'forecast': forecast[:, i].tolist(),
                'mean': float(forecast[:, i].mean()),
                'std': float(forecast[:, i].std()),
                'trend': 'up' if forecast[-1, i] > forecast[0, i] else 'down'
            }
        return formatted

    def summarize_variance_decomposition(self, fevd):
        """Résume la décomposition de la variance"""
        # Simplifié pour l'exemple
        return {
            'note': 'La décomposition de la variance montre la contribution de chaque choc',
            'periods': 10,
            'summary': 'Disponible dans l\'objet fevd complet'
        }

    def calculate_granger_causality(self, var_model, columns):
        """Calcule la causalité de Granger"""
        gc_results = {}

        for cause in columns:
            for effect in columns:
                if cause != effect:
                    try:
                        test = var_model.test_causality(effect, [cause], kind='f')
                        gc_results[f"{cause}_causes_{effect}"] = {
                            'p_value': float(test.pvalue),
                            'significant': test.pvalue < 0.05
                        }
                    except:
                        continue

        # Filtrer les relations significatives
        significant = {k: v for k, v in gc_results.items() if v['significant']}

        return {
            'all_tests': gc_results,
            'significant_causality': significant,
            'count_significant': len(significant)
        }

    def identify_significant_vars(self, var_model):
        """Identifie les variables significatives dans le VAR"""
        significant = []

        try:
            summary = var_model.summary()
            # Parsing simplifié
            for i, param in enumerate(var_model.params.index):
                if 'GDP' in param:
                    significant.append({'variable': param, 'impact': 'high'})
                elif 'CPI' in param:
                    significant.append({'variable': param, 'impact': 'high'})
        except:
            pass

        return significant

    def run_factor_analysis(self):
        """Exécute l'analyse factorielle (PCA)"""
        print("\n📈 EXÉCUTION DE L'ANALYSE FACTORIELLE (PCA)")
        print("-" * 40)

        try:
            # Préparer les données
            factor_data = self.prepare_factor_data()

            if factor_data is None or factor_data.shape[1] < 3:
                print("⚠️  Données insuffisantes pour PCA")
                return None

            # Standardiser
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(factor_data)

            # Déterminer le nombre optimal de composantes
            n_components = min(5, factor_data.shape[1])
            pca = PCA(n_components=n_components)
            pca_result = pca.fit(scaled_data)

            # Analyser les résultats
            factor_results = {
                'explained_variance': pca_result.explained_variance_ratio_.tolist(),
                'cumulative_variance': np.cumsum(pca_result.explained_variance_ratio_).tolist(),
                'n_components': n_components,
                'components': self.analyze_components(pca_result, factor_data.columns),
                'factor_interpretation': self.interpret_factors(pca_result, factor_data.columns),
                'variance_threshold': 0.8  # Seuil d'explication
            }

            print(f"✅ Analyse factorielle terminée")
            print(f"   Composantes: {n_components}")
            print(f"   Variance expliquée: {sum(pca_result.explained_variance_ratio_)*100:.1f}%")

            return factor_results

        except Exception as e:
            print(f"❌ Erreur analyse factorielle: {e}")
            return None

    def prepare_factor_data(self):
        """Prépare les données pour l'analyse factorielle"""
        # Combiner les indicateurs actuels
        indicators = []
        feature_names = []

        # Indicateurs économiques
        econ_data = self.macro_data.get('economic_indicators', {})
        for key, data in econ_data.items():
            if 'value' in data:
                indicators.append(data['value'])
                feature_names.append(key)

        # Données de marché
        market_data = self.macro_data.get('market_data', {})
        for key, data in market_data.items():
            if 'price' in data:
                indicators.append(data['price'])
                feature_names.append(f"MKT_{key}")

        if len(indicators) < 3:
            return None

        # Créer DataFrame
        df = pd.DataFrame([indicators], columns=feature_names)

        return df

    def analyze_components(self, pca_model, feature_names):
        """Analyse les composantes principales"""
        components = {}

        for i in range(pca_model.n_components_):
            component_weights = pca_model.components_[i]

            # Associer les poids aux features
            feature_weights = {}
            for j, feature in enumerate(feature_names):
                feature_weights[feature] = float(component_weights[j])

            # Trier par importance absolue
            sorted_weights = dict(sorted(
                feature_weights.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5])  # Top 5 features

            components[f'PC{i+1}'] = {
                'explained_variance': float(pca_model.explained_variance_ratio_[i]),
                'top_features': sorted_weights,
                'interpretation': self.interpret_component(sorted_weights)
            }

        return components

    def interpret_component(self, feature_weights):
        """Interprète une composante principale"""
        features = list(feature_weights.keys())
        weights = list(feature_weights.values())

        # Logique d'interprétation
        if any('CPI' in f for f in features) and any('INTEREST' in f for f in features):
            return 'Facteur Inflation/Politique monétaire'
        elif any('GDP' in f for f in features):
            return 'Facteur Croissance économique'
        elif any('UNEMPLOYMENT' in f for f in features):
            return 'Facteur Marché du travail'
        elif any('VIX' in f for f in features) or any('SENTIMENT' in f for f in features):
            return 'Facteur Sentiment/Risque'
        elif any('GOLD' in f for f in features) or any('OIL' in f for f in features):
            return 'Facteur Matières premières'
        else:
            return 'Facteur économique composite'

    def interpret_factors(self, pca_model, feature_names):
        """Interprète l'ensemble des facteurs"""
        interpretations = []

        for i in range(pca_model.n_components_):
            var_exp = pca_model.explained_variance_ratio_[i]

            if var_exp > 0.15:  # Facteur significatif
                interpretations.append({
                    'factor': i + 1,
                    'variance_explained': float(var_exp * 100),
                    'importance': 'high' if var_exp > 0.25 else 'medium' if var_exp > 0.15 else 'low'
                })

        return interpretations

    def analyze_market_regime(self):
        """Identifie le régime de marché actuel"""
        print("\n🔄 IDENTIFICATION DU RÉGIME DE MARCHÉ")
        print("-" * 40)

        # Récupérer les indicateurs clés
        gdp_growth = self.macro_data['economic_indicators'].get('GDP_GROWTH', {}).get('value', 2.5)
        cpi_inflation = self.macro_data['economic_indicators'].get('CPI_INFLATION', {}).get('value', 3.0)
        unemployment = self.macro_data['economic_indicators'].get('UNEMPLOYMENT_RATE', {}).get('value', 3.8)

        # Données de marché
        vix = self.macro_data['market_data'].get('VIX', {}).get('price', 15.0)
        sp500_ret = self.macro_data['market_data'].get('SP500', {}).get('change_pct', 0)

        # Politique monétaire
        fed_stance = self.macro_data['monetary_policy'].get('FED_STANCE', {}).get('value', 'neutral')

        # Calculer les scores
        growth_score = self.score_growth(gdp_growth, unemployment)
        inflation_score = self.score_inflation(cpi_inflation, fed_stance)
        risk_score = self.score_risk(vix, sp500_ret)

        # Déterminer le régime
        regime = self.determine_regime(growth_score, inflation_score, risk_score)

        regime_analysis = {
            'current_regime': regime,
            'scores': {
                'growth': growth_score,
                'inflation': inflation_score,
                'risk': risk_score
            },
            'indicators': {
                'gdp_growth': gdp_growth,
                'cpi_inflation': cpi_inflation,
                'vix': vix,
                'fed_stance': fed_stance
            },
            'confidence': self.calculate_regime_confidence(growth_score, inflation_score, risk_score),
            'duration_estimate': self.estimate_regime_duration(regime)
        }

        print(f"✅ Régime identifié: {regime.upper()}")

        return regime_analysis

    def score_growth(self, gdp_growth, unemployment):
        """Score la croissance économique"""
        if gdp_growth > 3.0 and unemployment < 4.0:
            return 2  # Forte croissance
        elif gdp_growth > 2.0 and unemployment < 5.0:
            return 1  # Croissance modérée
        elif gdp_growth < 1.0:
            return -1  # Croissance faible
        else:
            return 0  # Neutre

    def score_inflation(self, cpi_inflation, fed_stance):
        """Score l'inflation"""
        if cpi_inflation > 4.0 and fed_stance == 'hawkish':
            return -2  # Inflation élevée, réponse agressive
        elif cpi_inflation > 3.0 and fed_stance == 'hawkish':
            return -1  # Inflation modérée, réponse modérée
        elif cpi_inflation < 2.0 and fed_stance == 'dovish':
            return 1   # Inflation basse, politique accommodante
        else:
            return 0   # Neutre

    def score_risk(self, vix, sp500_return):
        """Score le risque de marché"""
        if vix > 20 and sp500_return < -5:
            return -2  # Haut risque, marché baissier
        elif vix > 15 and sp500_return < 0:
            return -1  # Risque modéré, marché faible
        elif vix < 15 and sp500_return > 5:
            return 1   # Faible risque, marché haussier
        else:
            return 0   # Neutre

    def determine_regime(self, growth_score, inflation_score, risk_score):
        """Détermine le régime basé sur les scores"""
        total_score = growth_score + inflation_score + risk_score

        if total_score >= 3:
            return "expansion_robuste"
        elif total_score >= 1:
            return "expansion_moderee"
        elif total_score <= -3:
            return "recession"
        elif total_score <= -1:
            return "ralentissement"
        elif inflation_score <= -2:
            return "stagflation"
        elif risk_score <= -2:
            return "crise_risque"
        else:
            return "transition"

    def calculate_regime_confidence(self, growth_score, inflation_score, risk_score):
        """Calcule la confiance dans l'identification du régime"""
        scores = [abs(growth_score), abs(inflation_score), abs(risk_score)]
        max_score = max(scores)

        if max_score >= 2:
            return "haute"
        elif max_score >= 1:
            return "moyenne"
        else:
            return "faible"

    def estimate_regime_duration(self, regime):
        """Estime la durée probable du régime"""
        duration_map = {
            'expansion_robuste': '6-18 mois',
            'expansion_moderee': '3-12 mois',
            'ralentissement': '2-6 mois',
            'recession': '6-18 mois',
            'stagflation': '12-24 mois',
            'crise_risque': '1-3 mois',
            'transition': '1-3 mois'
        }
        return duration_map.get(regime, 'inconnu')

    def generate_macro_recommendations(self, regime_analysis):
        """Génère des recommandations basées sur l'analyse macro"""
        print("\n💡 GÉNÉRATION DES RECOMMANDATIONS MACRO")
        print("-" * 40)

        regime = regime_analysis['current_regime']
        horizon = self.user_profile['analysis_horizon']
        risk_tolerance = self.user_profile['risk_tolerance']

        recommendations = {
            'strategic_view': self.get_strategic_view(regime),
            'asset_allocation': self.get_asset_allocation(regime, risk_tolerance),
            'sector_recommendations': self.get_sector_recommendations(regime),
            'geographic_exposure': self.get_geographic_exposure(),
            'risk_management': self.get_risk_management_recommendations(regime, risk_tolerance),
            'monitoring_signals': self.get_monitoring_signals(),
            'actionable_insights': self.get_actionable_insights(regime, horizon)
        }

        return recommendations

    def get_strategic_view(self, regime):
        """Retourne la vue stratégique basée sur le régime"""
        views = {
            'expansion_robuste': {
                'view': 'HAUSSIER',
                'rationale': 'Croissance forte, inflation contrôlée, politique accommodante',
                'key_message': 'Favoriser les actifs risqués, réduire les hedge'
            },
            'expansion_moderee': {
                'view': 'MODÉRÉMENT HAUSSIER',
                'rationale': 'Croissance positive mais modérée, vigilance sur l\'inflation',
                'key_message': 'Positionnement sélectif, diversification maintenue'
            },
            'ralentissement': {
                'view': 'NEUTRE À PRUDENT',
                'rationale': 'Ralentissement de la croissance, risques accrus',
                'key_message': 'Réduire l\'exposition risque, augmenter la qualité'
            },
            'recession': {
                'view': 'BAISSIER',
                'rationale': 'Contraction économique, chômage en hausse',
                'key_message': 'Défensif, liquidité, obligations longues'
            },
            'stagflation': {
                'view': 'DIFFICILE',
                'rationale': 'Inflation élevée avec croissance faible',
                'key_message': 'Actifs réels, court terme, éviter les obligations longues'
            },
            'crise_risque': {
                'view': 'DÉFENSIF',
                'rationale': 'Stress de marché, volatilité élevée',
                'key_message': 'Liquidité, hedging, attendre clarification'
            },
            'transition': {
                'view': 'ATTENTISME',
                'rationale': 'Signaux contradictoires, changement de régime probable',
                'key_message': 'Positionnement neutre, attendre confirmation'
            }
        }

        return views.get(regime, {
            'view': 'NEUTRE',
            'rationale': 'Données insuffisantes pour déterminer une vue claire',
            'key_message': 'Maintenir la diversification'
        })

    def get_asset_allocation(self, regime, risk_tolerance):
        """Retourne l'allocation d'actifs recommandée"""

        # Modèles d'allocation par régime et tolérance au risque
        allocation_models = {
            'expansion_robuste': {
                'conservateur': {'Equities': 50, 'Bonds': 40, 'Commodities': 5, 'Cash': 5},
                'modéré': {'Equities': 65, 'Bonds': 25, 'Commodities': 5, 'Cash': 5},
                'agressif': {'Equities': 80, 'Bonds': 15, 'Commodities': 5, 'Cash': 0}
            },
            'expansion_moderee': {
                'conservateur': {'Equities': 45, 'Bonds': 45, 'Commodities': 5, 'Cash': 5},
                'modéré': {'Equities': 60, 'Bonds': 30, 'Commodities': 5, 'Cash': 5},
                'agressif': {'Equities': 70, 'Bonds': 20, 'Commodities': 5, 'Cash': 5}
            },
            'ralentissement': {
                'conservateur': {'Equities': 35, 'Bonds': 50, 'Commodities': 5, 'Cash': 10},
                'modéré': {'Equities': 50, 'Bonds': 40, 'Commodities': 5, 'Cash': 5},
                'agressif': {'Equities': 60, 'Bonds': 30, 'Commodities': 5, 'Cash': 5}
            },
            'recession': {
                'conservateur': {'Equities': 25, 'Bonds': 60, 'Gold': 10, 'Cash': 5},
                'modéré': {'Equities': 35, 'Bonds': 50, 'Gold': 10, 'Cash': 5},
                'agressif': {'Equities': 45, 'Bonds': 40, 'Gold': 10, 'Cash': 5}
            },
            'stagflation': {
                'conservateur': {'Equities': 30, 'Bonds': 30, 'Commodities': 25, 'Cash': 15},
                'modéré': {'Equities': 40, 'Bonds': 25, 'Commodities': 25, 'Cash': 10},
                'agressif': {'Equities': 50, 'Bonds': 20, 'Commodities': 25, 'Cash': 5}
            }
        }

        default_alloc = {'Equities': 50, 'Bonds': 40, 'Cash': 10}

        regime_alloc = allocation_models.get(regime, allocation_models.get('expansion_moderee', {}))
        allocation = regime_alloc.get(risk_tolerance, default_alloc)

        # Ajuster selon l'univers d'investissement
        investment_universe = self.user_profile.get('investment_universe', [])

        if 'commodities' not in investment_universe and 'Commodities' in allocation:
            # Redistribuer aux autres actifs
            comm_pct = allocation.pop('Commodities', 0)
            if 'Equities' in allocation:
                allocation['Equities'] += comm_pct / 2
            if 'Bonds' in allocation:
                allocation['Bonds'] += comm_pct / 2

        return allocation

    def get_sector_recommendations(self, regime):
        """Retourne les recommandations sectorielles"""

        sector_views = {
            'expansion_robuste': {
                'OVERWEIGHT': ['Technology', 'Consumer Discretionary', 'Financials', 'Industrials'],
                'NEUTRAL': ['Communication Services', 'Materials', 'Real Estate'],
                'UNDERWEIGHT': ['Utilities', 'Consumer Staples', 'Healthcare']
            },
            'expansion_moderee': {
                'OVERWEIGHT': ['Technology', 'Healthcare', 'Financials'],
                'NEUTRAL': ['Consumer Discretionary', 'Industrials', 'Communication Services'],
                'UNDERWEIGHT': ['Utilities', 'Real Estate', 'Energy']
            },
            'ralentissement': {
                'OVERWEIGHT': ['Healthcare', 'Consumer Staples', 'Utilities'],
                'NEUTRAL': ['Technology', 'Communication Services', 'Financials'],
                'UNDERWEIGHT': ['Consumer Discretionary', 'Industrials', 'Materials']
            },
            'recession': {
                'OVERWEIGHT': ['Consumer Staples', 'Utilities', 'Healthcare'],
                'NEUTRAL': ['Communication Services', 'Technology'],
                'UNDERWEIGHT': ['Financials', 'Industrials', 'Consumer Discretionary']
            },
            'stagflation': {
                'OVERWEIGHT': ['Energy', 'Materials', 'Consumer Staples'],
                'NEUTRAL': ['Healthcare', 'Utilities'],
                'UNDERWEIGHT': ['Technology', 'Consumer Discretionary', 'Real Estate']
            }
        }

        return sector_views.get(regime, {
            'OVERWEIGHT': ['Technology', 'Healthcare'],
            'NEUTRAL': ['All other sectors'],
            'UNDERWEIGHT': []
        })

    def get_geographic_exposure(self):
        """Retourne les recommandations d'exposition géographique"""
        country = self.user_profile['country_focus']

        exposures = {
            'US': {
                'Domestic': 60,
                'Developed ex-US': 25,
                'Emerging Markets': 15
            },
            'EU': {
                'Domestic': 50,
                'US': 30,
                'Other Developed': 15,
                'Emerging Markets': 5
            },
            'CN': {
                'Domestic': 40,
                'Developed Markets': 40,
                'Other Emerging': 20
            },
            'JP': {
                'Domestic': 40,
                'US': 30,
                'Other Developed': 20,
                'Emerging Markets': 10
            },
            'EM': {
                'Domestic': 30,
                'Developed Markets': 50,
                'Other Emerging': 20
            },
            'GLOBAL': {
                'US': 40,
                'Developed ex-US': 30,
                'Emerging Markets': 30
            }
        }

        return exposures.get(country, {'Global Diversified': 100})

    def get_risk_management_recommendations(self, regime, risk_tolerance):
        """Retourne les recommandations de gestion du risque"""

        recommendations = []

        # Volatilité attendue
        vix = self.macro_data['market_data'].get('VIX', {}).get('price', 15)
        if vix > 20:
            recommendations.append({
                'action': 'Augmenter hedging',
                'instruments': ['VIX calls', 'Put spreads', 'Gold'],
                'allocation': '5-10%'
            })

        # Risque géopolitique
        geo_risk = self.macro_data['geopolitical'].get('GLOBAL_RISK_INDEX', {}).get('value', 0.35)
        if geo_risk > 0.4:
            recommendations.append({
                'action': 'Ajouter hedge géopolitique',
                'instruments': ['Gold (GLD)', 'Swiss Franc', 'Defense stocks'],
                'allocation': '3-5%'
            })

        # Inflation
        cpi = self.macro_data['economic_indicators'].get('CPI_INFLATION', {}).get('value', 3.0)
        if cpi > 3.5:
            recommendations.append({
                'action': 'Protection inflation',
                'instruments': ['TIPs', 'Commodities', 'Real Estate'],
                'allocation': '5-15%'
            })

        # Recommandations par régime
        if regime in ['recession', 'crise_risque']:
            recommendations.append({
                'action': 'Augmenter liquidité',
                'instruments': ['Cash', 'Money Market', 'Short-term Treasuries'],
                'allocation': '10-20%'
            })

        if len(recommendations) == 0:
            recommendations.append({
                'action': 'Maintenir hedges existants',
                'rationale': 'Environnement de risque stable'
            })

        return recommendations

    def get_monitoring_signals(self):
        """Retourne les signaux à surveiller"""

        signals = []

        # Signaux économiques
        signals.append({
            'type': 'ECONOMIC',
            'signal': 'Prochaine publication CPI',
            'threshold': '> 3.5% (hawkish) ou < 2.5% (dovish)',
            'impact': 'HAUT'
        })

        signals.append({
            'type': 'ECONOMIC',
            'signal': 'Rapport emploi US',
            'threshold': 'Taux chômage > 4.0%',
            'impact': 'MOYEN'
        })

        # Signaux de marché
        vix = self.macro_data['market_data'].get('VIX', {}).get('price', 15)
        if vix < 12:
            signals.append({
                'type': 'MARKET',
                'signal': 'VIX très bas (complaisance)',
                'threshold': 'VIX > 20',
                'impact': 'MOYEN'
            })

        # Signaux géopolitiques
        geo_hotspots = self.macro_data['geopolitical'].get('GLOBAL_RISK_INDEX', {}).get('hotspots', [])
        if geo_hotspots:
            signals.append({
                'type': 'GEOPOLITICAL',
                'signal': f'Tensions: {", ".join(geo_hotspots[:2])}',
                'threshold': 'Escalade significative',
                'impact': 'HAUT'
            })

        return signals

    def get_actionable_insights(self, regime, horizon):
        """Retourne les insights actionnables"""

        insights = []

        # Insights généraux
        insights.append({
            'priority': 'HIGH',
            'action': 'Revoir allocation selon nouveau régime',
            'timing': 'Immédiat',
            'details': f'Adapter l\'allocation au régime {regime}'
        })

        # Insights par horizon
        if horizon == 'court':
            insights.append({
                'priority': 'MEDIUM',
                'action': 'Surveiller données économiques hebdomadaires',
                'timing': 'Chaque vendredi',
                'details': 'CPI, emploi, ventes au détail'
            })
        elif horizon == 'moyen':
            insights.append({
                'priority': 'MEDIUM',
                'action': 'Préparer rotation sectorielle',
                'timing': '1-3 mois',
                'details': 'Se positionner pour le prochain cycle'
            })
        else:  # long
            insights.append({
                'priority': 'LOW',
                'action': 'Planifier investissements stratégiques',
                'timing': '6-12 mois',
                'details': 'Thèmes structurels: transition énergétique, digitalisation'
            })

        # Insights spécifiques au régime
        if regime == 'expansion_robuste':
            insights.append({
                'priority': 'HIGH',
                'action': 'Augmenter exposition actions',
                'timing': 'Prochain point bas',
                'details': 'Favoriser croissance et cycliques'
            })
        elif regime == 'recession':
            insights.append({
                'priority': 'HIGH',
                'action': 'Renforcer position défensive',
                'timing': 'Immédiat',
                'details': 'Augmenter qualité, liquidité, hedging'
            })

        return insights

    def generate_final_report(self, var_results, factor_results, regime_analysis, recommendations):
        """Génère le rapport final d'analyse macro"""

        report = {
            'executive_summary': self.generate_executive_summary(regime_analysis, recommendations),
            'user_profile': self.user_profile,
            'data_overview': {
                'economic_indicators_count': len(self.macro_data.get('economic_indicators', {})),
                'market_data_count': len(self.macro_data.get('market_data', {})),
                'last_updated': self.macro_data.get('last_updated', 'N/A')
            },
            'regime_analysis': regime_analysis,
            'quantitative_analysis': {
                'var_model': var_results,
                'factor_model': factor_results
            },
            'recommendations': recommendations,
            'risk_assessment': self.assess_overall_risk(),
            'next_steps': self.suggest_next_steps()
        }

        return report

    def generate_executive_summary(self, regime_analysis, recommendations):
        """Génère le résumé exécutif"""

        regime = regime_analysis['current_regime']
        strategic_view = recommendations['strategic_view']

        return {
            'current_assessment': f"Régime de marché: {regime.upper()}",
            'strategic_view': strategic_view['view'],
            'key_message': strategic_view['key_message'],
            'confidence_level': regime_analysis['confidence'],
            'time_horizon': self.user_profile['analysis_horizon'],
            'primary_risks': self.identify_primary_risks(),
            'main_opportunities': self.identify_main_opportunities(regime)
        }

    def identify_primary_risks(self):
        """Identifie les principaux risques"""

        risks = []

        # Inflation
        cpi = self.macro_data['economic_indicators'].get('CPI_INFLATION', {}).get('value', 3.0)
        if cpi > 3.5:
            risks.append({
                'risk': 'Inflation persistante',
                'probability': 'MOYENNE',
                'impact': 'HAUT'
            })

        # Croissance
        gdp = self.macro_data['economic_indicators'].get('GDP_GROWTH', {}).get('value', 2.5)
        if gdp < 1.5:
            risks.append({
                'risk': 'Ralentissement économique',
                'probability': 'MOYENNE',
                'impact': 'HAUT'
            })

        # Géopolitique
        geo_risk = self.macro_data['geopolitical'].get('GLOBAL_RISK_INDEX', {}).get('value', 0.35)
        if geo_risk > 0.4:
            risks.append({
                'risk': 'Tensions géopolitiques',
                'probability': 'ÉLEVÉE',
                'impact': 'MOYEN'
            })

        # Politique monétaire
        fed_stance = self.macro_data['monetary_policy'].get('FED_STANCE', {}).get('value', 'neutral')
        if fed_stance == 'hawkish':
            risks.append({
                'risk': 'Reserrement monétaire',
                'probability': 'MOYENNE',
                'impact': 'MOYEN'
            })

        return risks if risks else [{'risk': 'Aucun risque majeur identifié', 'probability': 'FAIBLE', 'impact': 'FAIBLE'}]

    def identify_main_opportunities(self, regime):
        """Identifie les principales opportunités"""

        opportunities = []

        if regime in ['expansion_robuste', 'expansion_moderee']:
            opportunities.append({
                'opportunity': 'Actions croissance',
                'rationale': 'Environnement favorable à la croissance des bénéfices',
                'sectors': ['Technology', 'Consumer Discretionary']
            })

        if regime == 'recession':
            opportunities.append({
                'opportunity': 'Obligations long terme',
                'rationale': 'Taux susceptibles de baisser en récession',
                'instruments': ['Long-term Treasuries', 'Corporate bonds']
            })

        if self.macro_data['economic_indicators'].get('CPI_INFLATION', {}).get('value', 3.0) < 2.5:
            opportunities.append({
                'opportunity': 'Actions valeur',
                'rationale': 'Inflation basse favorable aux actions à dividendes',
                'sectors': ['Utilities', 'Consumer Staples']
            })

        return opportunities if opportunities else [
            {'opportunity': 'Diversification', 'rationale': 'Approche équilibrée dans environnement incertain'}
        ]

    def assess_overall_risk(self):
        """Évalue le risque global"""

        risk_score = 0
        risk_factors = []

        # Facteurs de risque
        vix = self.macro_data['market_data'].get('VIX', {}).get('price', 15)
        if vix > 20:
            risk_score += 2
            risk_factors.append('Volatilité élevée')

        geo_risk = self.macro_data['geopolitical'].get('GLOBAL_RISK_INDEX', {}).get('value', 0.35)
        if geo_risk > 0.4:
            risk_score += 2
            risk_factors.append('Risque géopolitique')

        cpi = self.macro_data['economic_indicators'].get('CPI_INFLATION', {}).get('value', 3.0)
        if cpi > 3.5:
            risk_score += 1
            risk_factors.append('Inflation élevée')

        # Niveau de risque
        if risk_score >= 4:
            level = 'ÉLEVÉ'
        elif risk_score >= 2:
            level = 'MODÉRÉ'
        else:
            level = 'FAIBLE'

        return {
            'risk_score': risk_score,
            'risk_level': level,
            'risk_factors': risk_factors,
            'recommendation': 'Augmenter hedging' if level == 'ÉLEVÉ' else 'Maintenir position'
        }

    def suggest_next_steps(self):
        """Suggère les prochaines étapes"""

        return [
            {
                'step': 1,
                'action': 'Implémenter les recommandations d\'allocation',
                'timeline': '1-2 semaines'
            },
            {
                'step': 2,
                'action': 'Mettre en place les hedges recommandés',
                'timeline': 'Immédiat'
            },
            {
                'step': 3,
                'action': 'Surveiller les signaux clés identifiés',
                'timeline': 'Continue'
            },
            {
                'step': 4,
                'action': 'Revoir l\'analyse dans 1 mois ou si changement majeur',
                'timeline': '1 mois'
            }
        ]

    def display_report(self, report):
        """Affiche le rapport de manière lisible"""

        print("\n" + "="*80)
        print("📊 MACRO-VISION: RAPPORT D'ANALYSE MACROÉCONOMIQUE COMPLET")
        print("="*80)

        # Résumé exécutif
        exec_summary = report['executive_summary']
        print(f"\n🎯 RÉSUMÉ EXÉCUTIF")
        print("-" * 40)
        print(f"📈 Régime de marché: {exec_summary['current_assessment']}")
        print(f"🎯 Vue stratégique: {exec_summary['strategic_view']}")
        print(f"💡 Message clé: {exec_summary['key_message']}")
        print(f"📊 Confiance: {exec_summary['confidence_level'].upper()}")

        # Vue d'ensemble des données
        print(f"\n📊 DONNÉES ANALYSÉES")
        print("-" * 40)
        data_info = report['data_overview']
        print(f"• Indicateurs économiques: {data_info['economic_indicators_count']}")
        print(f"• Instruments de marché: {data_info['market_data_count']}")
        print(f"• Dernière mise à jour: {data_info['last_updated'][:10]}")

        # Analyse du régime
        regime_info = report['regime_analysis']
        print(f"\n🔄 ANALYSE DU RÉGIME")
        print("-" * 40)
        print(f"• Régime actuel: {regime_info['current_regime'].upper()}")
        print(f"• Durée estimée: {regime_info['duration_estimate']}")
        print(f"• Scores: Croissance={regime_info['scores']['growth']}, "
              f"Inflation={regime_info['scores']['inflation']}, "
              f"Risque={regime_info['scores']['risk']}")

        # Recommandations principales
        recs = report['recommendations']
        print(f"\n💡 RECOMMANDATIONS PRINCIPALES")
        print("-" * 40)

        # Allocation d'actifs
        print(f"💰 ALLOCATION D'ACTIFS:")
        allocation = recs['asset_allocation']
        for asset, pct in allocation.items():
            print(f"   • {asset}: {pct}%")

        # Vue stratégique
        print(f"\n🎯 VUE STRATÉGIQUE:")
        strategic = recs['strategic_view']
        print(f"   • {strategic['view']}")
        print(f"   • Raison: {strategic['rationale']}")

        # Recommandations sectorielles
        print(f"\n🏭 RECOMMANDATIONS SECTORIELLES:")
        sectors = recs['sector_recommendations']
        for weight, sector_list in sectors.items():
            if sector_list:
                print(f"   • {weight}: {', '.join(sector_list)}")

        # Gestion du risque
        print(f"\n⚠️  GESTION DU RISQUE:")
        risk_mgmt = recs['risk_management']
        for rec in risk_mgmt[:2]:  # Afficher les 2 premières recommandations
            print(f"   • {rec['action']}")

        # Insights actionnables
        print(f"\n🚀 INSIGHTS ACTIONNABLES:")
        insights = recs['actionable_insights']
        for insight in insights[:2]:
            print(f"   • [{insight['priority']}] {insight['action']}")

        # Analyse quantitative
        print(f"\n🔬 ANALYSE QUANTITATIVE")
        print("-" * 40)
        quant = report['quantitative_analysis']

        if quant['var_model']:
            print(f"• Analyse VAR: Modèle à {quant['var_model']['optimal_lags']} lags")
            print(f"  AIC: {quant['var_model']['model_metrics']['aic']:.1f}")

        if quant['factor_model']:
            print(f"• Analyse factorielle: {quant['factor_model']['n_components']} facteurs")
            print(f"  Variance expliquée: {sum(quant['factor_model']['explained_variance'])*100:.1f}%")

        # Évaluation du risque
        print(f"\n📉 ÉVALUATION DU RISQUE GLOBAL")
        print("-" * 40)
        risk_assessment = report['risk_assessment']
        print(f"• Niveau de risque: {risk_assessment['risk_level']}")
        print(f"• Score: {risk_assessment['risk_score']}/6")
        if risk_assessment['risk_factors']:
            print(f"• Facteurs: {', '.join(risk_assessment['risk_factors'])}")

        # Prochaines étapes
        print(f"\n📋 PROCHAINES ÉTAPES")
        print("-" * 40)
        next_steps = report['next_steps']
        for step in next_steps:
            print(f"{step['step']}. {step['action']} ({step['timeline']})")

        print("\n" + "="*80)
        print("✅ ANALYSE MACROÉCONOMIQUE TERMINÉE")
        print("="*80)

    def run_complete_analysis(self):
        """Exécute l'analyse macroéconomique complète"""

        print("\n" + "="*80)
        print("🌍 MACRO-VISION: AGENT D'ANALYSE MACROÉCONOMIQUE INDÉPENDANT")
        print("="*80)

        # 1. Configuration utilisateur
        self.get_user_inputs()

        # 2. Collecte des données
        self.fetch_economic_data()

        # 3. Analyses quantitatives
        print("\n" + "="*80)
        print("🔬 ANALYSES QUANTITATIVES EN COURS")
        print("="*80)

        var_results = self.run_var_analysis()
        factor_results = self.run_factor_analysis()

        # 4. Identification du régime
        regime_analysis = self.analyze_market_regime()

        # 5. Génération des recommandations
        recommendations = self.generate_macro_recommendations(regime_analysis)

        # 6. Génération du rapport
        report = self.generate_final_report(var_results, factor_results, regime_analysis, recommendations)

        # 7. Affichage du rapport
        self.display_report(report)

        # 8. Option de sauvegarde
        self.offer_report_save(report)

        return report

    def offer_report_save(self, report):
        """Propose de sauvegarder le rapport"""

        save_choice = input("\n💾 Souhaitez-vous sauvegarder ce rapport? (oui/non) [oui]: ").strip().lower()

        if not save_choice or save_choice.startswith('o'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"macro_vision_report_{timestamp}.json"

            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                print(f"✅ Rapport sauvegardé: {filename}")
            except Exception as e:
                print(f"❌ Erreur sauvegarde: {e}")

        # Option pour nouvelle analyse
        new_analysis = input("\n🔄 Effectuer une nouvelle analyse? (oui/non) [non]: ").strip().lower()
        if new_analysis.startswith('o'):
            self.__init__()  # Réinitialiser
            self.run_complete_analysis()

# Interface principale
def main():
    """Point d'entrée principal"""

    print("\n" + "="*80)
    print("🌐 MACRO-VISION: ANALYSE MACROÉCONOMIQUE AVANCÉE")
    print("="*80)
    print("\nCet agent analyse indépendamment les données macroéconomiques pour")
    print("générer des recommandations de trading basées sur:")
    print("• Régimes de marché")
    print("• Cycles économiques")
    print("• Politiques monétaires")
    print("• Risques géopolitiques")
    print("• Modèles VAR et analyse factorielle")

    agent = MacroVisionAgent()

    try:
        agent.run_complete_analysis()
    except KeyboardInterrupt:
        print("\n\n❌ Analyse interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        print("\nVeuillez réessayer")

if __name__ == "__main__":
    main()