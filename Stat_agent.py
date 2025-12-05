# trading_agent_portfolio_optimized.py - Agent de Trading Optimisé avec Gestion de Portefeuille

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')

# Modèles statistiques
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from scipy import stats
import concurrent.futures

class PortfolioTradingAgent:
    """
    Agent de trading avec gestion de portefeuille personnel
    et recommandations cross-market - VERSION OPTIMISÉE
    Compatible with MultiAgentOrchestrator in the file "Orchestrator.py" .
    """

    def __init__(self):
        self.user_portfolio = {}  # Format: {symbol: {'shares': qty, 'avg_price': price}}
        self.user_cash = 0.0
        self.risk_profile = 'moderate'
        self.investment_horizon = 'medium'  # short/medium/long
        self.market_preferences = []  # Secteurs préférés
        self.excluded_sectors = []   # Secteurs exclus

        # CACHE pour optimisation
        self._data_cache = {}
        self._sector_cache = {}
        self._analysis_cache = {}

    def setup_user_profile(self):
        """Configuration initiale du profil utilisateur"""
        print("\n" + "="*60)
        print("👤 CONFIGURATION DU PROFIL UTILISATEUR")
        print("="*60)

        # Profil de risque
        self.risk_profile = input(
            "Profil de risque (conservateur/moderé/agressif) [modéré]: "
        ).strip().lower() or 'modéré'

        # Horizon d'investissement
        self.investment_horizon = input(
            "Horizon d'investissement (court/moyen/long terme) [moyen]: "
        ).strip().lower() or 'moyen'

        # Cash disponible
        cash_input = input(
            "Cash disponible pour nouveaux investissements ($) [0]: "
        ).strip()
        self.user_cash = float(cash_input) if cash_input else 0.0

        # Préférences de marché
        print("\n💼 Préférences de marché (séparer par des virgules):")
        print("Ex: technologie, énergie, santé, crypto, immobilier, matières premières")
        prefs = input("Secteurs préférés: ").strip()
        if prefs:
            self.market_preferences = [p.strip().lower() for p in prefs.split(',')]

        # Exclusions
        exclusions = input("Secteurs à exclure: ").strip()
        if exclusions:
            self.excluded_sectors = [e.strip().lower() for e in exclusions.split(',')]

        return self

    def input_current_portfolio(self):
        """Saisie du portefeuille actuel de l'utilisateur"""
        print("\n" + "="*60)
        print("📊 SAISIE DU PORTEFEUILLE ACTUEL")
        print("="*60)
        print("Format: SYMBOLE QUANTITÉ PRIX_MOYEN (ex: AAPL 10 150.50)")
        print("Laissez vide pour terminer\n")

        self.user_portfolio = {}

        while True:
            entry = input("Position (SYM QTY AVG_PRICE) ou 'fin': ").strip()

            if entry.lower() in ['fin', '', 'done']:
                break

            try:
                parts = entry.split()
                if len(parts) >= 2:
                    symbol = parts[0].upper()
                    shares = float(parts[1])
                    avg_price = float(parts[2]) if len(parts) > 2 else 0.0

                    if symbol in self.user_portfolio:
                        # Fusion des positions existantes
                        total_shares = self.user_portfolio[symbol]['shares'] + shares
                        total_cost = (self.user_portfolio[symbol]['shares'] *
                                    self.user_portfolio[symbol]['avg_price'] +
                                    shares * avg_price)
                        new_avg = total_cost / total_shares if total_shares > 0 else 0

                        self.user_portfolio[symbol] = {
                            'shares': total_shares,
                            'avg_price': new_avg
                        }
                        print(f"✓ Position {symbol} mise à jour")
                    else:
                        self.user_portfolio[symbol] = {
                            'shares': shares,
                            'avg_price': avg_price
                        }
                        print(f"✓ Position {symbol} ajoutée")
                else:
                    print("Format invalide")

            except Exception as e:
                print(f"Erreur: {e}")

        print(f"\n✅ Portefeuille enregistré: {len(self.user_portfolio)} positions")
        return self

    def get_cached_data(self, symbol: str):
        """Récupère ou met en cache les données"""
        if symbol in self._data_cache:
            cache_time, data = self._data_cache[symbol]
            # Cache valide 5 minutes
            if (datetime.now() - cache_time).seconds < 300:
                return data

        # Nouvelle récupération
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1mo", interval="1d", prepost=False)
            info = ticker.info

            data = (hist, info)
            self._data_cache[symbol] = (datetime.now(), data)

            return data
        except:
            return (pd.DataFrame(), {})

    def analyze_user_portfolio(self):
        """Analyse complète du portefeuille utilisateur - OPTIMISÉ"""
        if not self.user_portfolio:
            print("⚠️  Portefeuille vide")
            return None

        print("\n" + "="*60)
        print("🔍 ANALYSE DU PORTEFEUILLE PERSONNEL")
        print("="*60)

        portfolio_value = 0
        portfolio_cost = 0
        positions_analysis = {}

        # LIMITER le nombre de workers et ajouter timeout
        max_workers = min(3, len(self.user_portfolio))  # Max 3 threads
        timeout_per_symbol = 15  # Secondes max par symbole

        print(f"⏳ Analyse de {len(self.user_portfolio)} positions...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}

            for symbol, position in self.user_portfolio.items():
                # Skip les symboles invalides rapidement
                if not symbol or len(symbol) < 2:
                    continue

                future = executor.submit(self.analyze_single_position, symbol, position)
                futures[future] = symbol

            completed = 0
            total = len(futures)

            for future in concurrent.futures.as_completed(futures, timeout=timeout_per_symbol * total):
                symbol = futures[future]
                completed += 1

                # Afficher progression
                print(f"\r🔍 Progression: {completed}/{total} ({completed/total*100:.0f}%)", end="")

                try:
                    result = future.result(timeout=10)
                    if result:
                        positions_analysis[symbol] = result

                        current_value = result['current_value']
                        cost_basis = result['cost_basis']
                        portfolio_value += current_value
                        portfolio_cost += cost_basis

                        # Afficher immédiatement
                        perf_pct = result['pnl_pct']
                        signal = result['signal']['decision']
                        print(f"\n  {symbol}: {signal} | Perf: {perf_pct:+.1f}% | "
                              f"Val: ${current_value:.0f}")

                except concurrent.futures.TimeoutError:
                    print(f"\n✗ {symbol}: Timeout - analyse trop longue")
                except Exception as e:
                    print(f"\n✗ {symbol}: Erreur - {str(e)[:50]}")

        print()  # Nouvelle ligne après la progression

        # Vérifier si on a des résultats
        if not positions_analysis:
            print("❌ Aucune position analysée avec succès")
            return None

        # Analyse globale
        total_performance = ((portfolio_value - portfolio_cost) / portfolio_cost * 100) if portfolio_cost > 0 else 0

        portfolio_analysis = {
            'total_value': portfolio_value,
            'total_cost': portfolio_cost,
            'total_performance_pct': total_performance,
            'cash_available': self.user_cash,
            'total_assets': portfolio_value + self.user_cash,
            'positions': positions_analysis,
            'diversification': self.calculate_diversification(positions_analysis),
            'risk_metrics': self.calculate_portfolio_risk(positions_analysis)
        }

        self.display_portfolio_summary(portfolio_analysis)
        return portfolio_analysis

    def analyze_user_portfolio_fast(self):
        """Analyse RAPIDE du portefeuille utilisateur"""
        if not self.user_portfolio:
            print("⚠️  Portefeuille vide")
            return None

        print("\n" + "="*60)
        print("⚡ ANALYSE RAPIDE DU PORTEFEUILLE")
        print("="*60)

        portfolio_value = 0
        portfolio_cost = 0
        positions_analysis = {}

        print(f"⏳ Analyse rapide de {len(self.user_portfolio)} positions...")

        # Mode séquentiel pour plus de stabilité
        for i, (symbol, position) in enumerate(self.user_portfolio.items()):
            print(f"\r🔍 Progression: {i+1}/{len(self.user_portfolio)}", end="")

            try:
                result = self.analyze_single_position_fast(symbol, position)
                if result:
                    positions_analysis[symbol] = result

                    current_value = result['current_value']
                    cost_basis = result['cost_basis']
                    portfolio_value += current_value
                    portfolio_cost += cost_basis

                    perf_pct = result['pnl_pct']
                    signal = result['signal']['decision']
                    print(f"\n  {symbol}: {signal} | Perf: {perf_pct:+.1f}%")

            except Exception as e:
                print(f"\n✗ {symbol}: Erreur - {str(e)[:50]}")

        print()  # Nouvelle ligne

        if not positions_analysis:
            print("❌ Aucune position analysée avec succès")
            return None

        # Analyse globale
        total_performance = ((portfolio_value - portfolio_cost) / portfolio_cost * 100) if portfolio_cost > 0 else 0

        portfolio_analysis = {
            'total_value': portfolio_value,
            'total_cost': portfolio_cost,
            'total_performance_pct': total_performance,
            'cash_available': self.user_cash,
            'total_assets': portfolio_value + self.user_cash,
            'positions': positions_analysis,
            'diversification': self.calculate_diversification(positions_analysis),
            'risk_metrics': self.calculate_portfolio_risk(positions_analysis)
        }

        self.display_portfolio_summary(portfolio_analysis)
        return portfolio_analysis

    def analyze_single_position(self, symbol: str, position: Dict) -> Optional[Dict]:
        """Analyse d'une position individuelle - OPTIMISÉ"""
        try:
            # Récupération données via cache
            hist, info = self.get_cached_data(symbol)

            if hist.empty or len(hist) < 5:
                # Fallback rapide
                return self.analyze_single_position_fast(symbol, position)

            current_price = hist['Close'].iloc[-1]
            shares = position['shares']
            avg_price = position.get('avg_price', current_price)

            current_value = current_price * shares
            cost_basis = avg_price * shares
            unrealized_pnl = current_value - cost_basis
            pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

            # Analyse technique complète
            signal = self.quick_analysis(symbol, hist)

            # Recommandation personnalisée
            recommendation = self.generate_position_recommendation(
                symbol, signal, pnl_pct, shares, current_price, avg_price
            )

            # Info secteur
            sector = info.get('sector', 'Inconnu') if info else 'Inconnu'

            return {
                'symbol': symbol,
                'shares': shares,
                'avg_price': avg_price,
                'current_price': current_price,
                'current_value': current_value,
                'cost_basis': cost_basis,
                'unrealized_pnl': unrealized_pnl,
                'pnl_pct': pnl_pct,
                'signal': signal,
                'recommendation': recommendation,
                'sector': sector
            }

        except Exception as e:
            # Fallback sur version rapide en cas d'erreur
            print(f"  {symbol}: Analyse complète échouée, passage en mode rapide")
            return self.analyze_single_position_fast(symbol, position)

    def analyze_single_position_fast(self, symbol: str, position: Dict) -> Optional[Dict]:
        """Version rapide sans ARIMA/GARCH pour analyse initiale"""
        try:
            # Récupération minimale
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d", interval="1d")
            except:
                # Dernière tentative avec timeout
                return None

            if hist.empty or len(hist) < 2:
                return None

            current_price = hist['Close'].iloc[-1]
            shares = position['shares']
            avg_price = position.get('avg_price', current_price)

            current_value = current_price * shares
            cost_basis = avg_price * shares
            unrealized_pnl = current_value - cost_basis
            pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0

            # Analyse technique RAPIDE (sans ARIMA/GARCH)
            signal = self.quick_analysis_fast(symbol, hist)

            recommendation = self.generate_position_recommendation(
                symbol, signal, pnl_pct, shares, current_price, avg_price
            )

            return {
                'symbol': symbol,
                'shares': shares,
                'avg_price': avg_price,
                'current_price': current_price,
                'current_value': current_value,
                'cost_basis': cost_basis,
                'unrealized_pnl': unrealized_pnl,
                'pnl_pct': pnl_pct,
                'signal': signal,
                'recommendation': recommendation,
                'sector': 'À analyser'  # Déféré pour plus tard
            }

        except Exception as e:
            return None

    def quick_analysis(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Analyse rapide ARIMA-GARCH pour une position"""
        try:
            if len(data) < 20:
                return self.quick_analysis_fast(symbol, data)

            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()

            # ARIMA simple avec timeout implicite
            try:
                model = ARIMA(returns, order=(1,0,1))
                result = model.fit(method='css', disp=False)
                forecast = result.forecast(steps=3)[0]
                arima_signal = 1 if forecast.mean() > 0 else -1
            except:
                arima_signal = 0

            # GARCH simple
            try:
                if len(returns) > 30:
                    garch = arch_model(returns * 100, vol='Garch', p=1, q=1)
                    garch_fit = garch.fit(disp='off', show_warning=False)
                    forecast_vol = np.sqrt(garch_fit.forecast(horizon=3).variance.values[-1, :] / 10000)
                    is_high_vol = forecast_vol.mean() > returns.std() * 1.2
                else:
                    is_high_vol = returns.std() > 0.02  # Estimation simple
            except:
                is_high_vol = False

            # Signaux techniques
            current_price = data['Close'].iloc[-1]
            sma_20 = data['Close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else current_price
            rsi = self.calculate_rsi(data['Close'])

            # Décision combinée
            score = 0
            confidence = 60

            if arima_signal > 0:
                score += 0.3
            elif arima_signal < 0:
                score -= 0.3

            if current_price > sma_20:
                score += 0.2

            if rsi < 30:
                score += 0.2
            elif rsi > 70:
                score -= 0.2

            if is_high_vol:
                score *= 0.7  # Réduction en haute volatilité
                confidence = 50

            # Décision finale
            if score > 0.15:
                decision = "BUY"
                action = "Acheter plus"
            elif score < -0.15:
                decision = "SELL"
                action = "Vendre partiellement/totalement"
            else:
                decision = "HOLD"
                action = "Maintenir position"

            return {
                'decision': decision,
                'action': action,
                'score': score,
                'confidence': confidence,
                'reasons': [
                    f"Signal ARIMA: {'positif' if arima_signal > 0 else 'négatif' if arima_signal < 0 else 'neutre'}",
                    f"RSI: {rsi:.1f}",
                    f"Volatilité: {'élevée' if is_high_vol else 'normale'}"
                ]
            }

        except Exception as e:
            return self.quick_analysis_fast(symbol, data)

    def quick_analysis_fast(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Analyse ultra-rapide sans modèles lourds"""
        try:
            if len(data) < 2:
                return {'decision': 'HOLD', 'confidence': 50, 'reasons': ['Données insuffisantes']}

            current_price = data['Close'].iloc[-1]
            prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price

            # Simple momentum
            daily_return = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0

            # Calcul RSI rapide
            if len(data) >= 14:
                rsi = self.calculate_rsi_fast(data['Close'])
            else:
                rsi = 50

            # Décision basique
            if daily_return > 1.5 and rsi < 70:
                decision = "BUY"
                confidence = 65
                reason = "Hausse récente + RSI favorable"
            elif daily_return < -1.5 and rsi > 30:
                decision = "SELL"
                confidence = 65
                reason = "Baisse récente + RSI défavorable"
            elif rsi < 30:
                decision = "BUY"
                confidence = 60
                reason = "RSI bas (survente)"
            elif rsi > 70:
                decision = "SELL"
                confidence = 60
                reason = "RSI haut (surachat)"
            else:
                decision = "HOLD"
                confidence = 55
                reason = "Marché stable"

            return {
                'decision': decision,
                'action': f"{decision} - {reason}",
                'score': daily_return / 10,  # Normalisé
                'confidence': confidence,
                'reasons': [f"Variation: {daily_return:+.1f}%", f"RSI: {rsi:.0f}", reason]
            }

        except Exception as e:
            return {'decision': 'HOLD', 'confidence': 40, 'reasons': [f'Analyse rapide: {str(e)[:30]}']}

    def calculate_rsi_fast(self, prices: pd.Series) -> float:
        """Calcule le RSI de manière optimisée"""
        try:
            if len(prices) < 2:
                return 50

            delta = prices.diff()
            gain = delta.where(delta > 0, 0).mean()
            loss = -delta.where(delta < 0, 0).mean()

            if loss == 0:
                return 100 if gain > 0 else 50

            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return min(100, max(0, rsi))
        except:
            return 50

    def generate_position_recommendation(self, symbol: str, signal: Dict,
                                       pnl_pct: float, shares: float,
                                       current_price: float, avg_price: float) -> Dict:
        """Génère une recommandation personnalisée pour une position"""

        recommendation = {
            'action': 'HOLD',
            'percentage': 0,
            'reason': '',
            'target_price': None,
            'stop_loss': None
        }

        # Règles basées sur performance et signal
        if signal['decision'] == 'SELL' and pnl_pct > 20:
            recommendation['action'] = 'TAKE_PROFIT'
            recommendation['percentage'] = min(50, (pnl_pct - 15) * 2)  # Vendre plus si gros gain
            recommendation['reason'] = 'Prise de bénéfices recommandée (gain élevé + signal vente)'
            recommendation['target_price'] = current_price * 0.95
            recommendation['stop_loss'] = avg_price * 1.15  # Protéger les gains

        elif signal['decision'] == 'SELL' and pnl_pct < -10:
            recommendation['action'] = 'CUT_LOSSES'
            recommendation['percentage'] = 100 if pnl_pct < -20 else 50
            recommendation['reason'] = 'Limitation des pertes (performance faible + signal vente)'
            recommendation['target_price'] = None
            recommendation['stop_loss'] = current_price * 0.98

        elif signal['decision'] == 'BUY' and pnl_pct < 0:
            recommendation['action'] = 'AVERAGE_DOWN'
            recommendation['percentage'] = 25  # Ajouter 25% à la position
            recommendation['reason'] = 'Moyenne à la baisse (achat opportun)'
            recommendation['target_price'] = avg_price * 0.9
            recommendation['stop_loss'] = current_price * 0.85

        elif signal['decision'] == 'BUY' and pnl_pct > 0:
            recommendation['action'] = 'ADD_TO_WINNERS'
            recommendation['percentage'] = 15  # Ajouter modérément
            recommendation['reason'] = 'Renforcer les positions gagnantes'
            recommendation['target_price'] = current_price * 1.15
            recommendation['stop_loss'] = avg_price

        else:  # HOLD
            recommendation['action'] = 'HOLD'
            recommendation['reason'] = 'Maintenir, pas de signal fort'
            recommendation['target_price'] = current_price * 1.1 if pnl_pct > 0 else avg_price
            recommendation['stop_loss'] = current_price * 0.9 if pnl_pct > 0 else avg_price * 0.95

        return recommendation

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calcule le RSI"""
        if len(prices) < period + 1:
            return 50

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Éviter division par zéro
        loss = loss.replace(0, np.nan)
        rs = gain / loss

        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    def get_sector_info(self, symbol: str) -> str:
        """Récupère le secteur d'une action"""
        try:
            if symbol in self._sector_cache:
                return self._sector_cache[symbol]

            ticker = yf.Ticker(symbol)
            info = ticker.info
            sector = info.get('sector', 'Inconnu')
            self._sector_cache[symbol] = sector
            return sector
        except:
            return 'Inconnu'

    def calculate_diversification(self, positions_analysis: Dict) -> Dict:
        """Calcule la diversification du portefeuille"""
        sector_allocation = {}
        total_value = sum(pos['current_value'] for pos in positions_analysis.values() if pos)

        if total_value == 0:
            return {
                'sector_allocation': {},
                'num_sectors': 0,
                'diversification_score': 0,
                'concentration_risk': 'INCONNU'
            }

        for symbol, analysis in positions_analysis.items():
            if analysis:
                sector = analysis.get('sector', 'Inconnu')
                value = analysis['current_value']
                sector_allocation[sector] = sector_allocation.get(sector, 0) + value

        # Calcul concentration
        for sector in sector_allocation:
            sector_allocation[sector] = (sector_allocation[sector] / total_value) * 100

        # Score de diversification (0-100)
        num_sectors = len(sector_allocation)
        max_sector_pct = max(sector_allocation.values()) if sector_allocation else 0
        diversification_score = 100 - max_sector_pct

        return {
            'sector_allocation': sector_allocation,
            'num_sectors': num_sectors,
            'diversification_score': diversification_score,
            'concentration_risk': 'ÉLEVÉ' if max_sector_pct > 50 else 'MODÉRÉ' if max_sector_pct > 30 else 'FAIBLE'
        }

    def calculate_portfolio_risk(self, positions_analysis: Dict) -> Dict:
        """Calcule les métriques de risque du portefeuille"""
        if not positions_analysis:
            return {}

        total_value = sum(pos['current_value'] for pos in positions_analysis.values() if pos)

        # Volatilité estimée (simplifiée)
        volatilities = []
        for analysis in positions_analysis.values():
            if analysis:
                # Estimation basique de volatilité
                pnl_pct = abs(analysis['pnl_pct'])
                if pnl_pct > 0:
                    volatilities.append(min(pnl_pct / 10, 50))  # Normalisation

        avg_volatility = np.mean(volatilities) if volatilities else 20

        # Drawdown maximum (simulé)
        max_drawdown = min(-10, -avg_volatility * 2)

        # VaR simplifiée (95%)
        var_95 = -total_value * (avg_volatility / 100) * 1.645

        return {
            'estimated_volatility': avg_volatility,
            'max_drawdown_potential': max_drawdown,
            'var_95': var_95,
            'risk_level': 'ÉLEVÉ' if avg_volatility > 30 else 'MODÉRÉ' if avg_volatility > 15 else 'FAIBLE'
        }

    def display_portfolio_summary(self, portfolio_analysis: Dict):
        """Affiche le résumé du portefeuille"""
        print("\n" + "="*60)
        print("📈 RÉSUMÉ DU PORTEFEUILLE")
        print("="*60)

        print(f"\n💵 VALEUR TOTALE: ${portfolio_analysis['total_value']:,.0f}")
        print(f"📊 PERFORMANCE: {portfolio_analysis['total_performance_pct']:+.1f}%")
        print(f"💰 CASH DISPONIBLE: ${portfolio_analysis['cash_available']:,.0f}")
        print(f"🏦 ACTIFS TOTAUX: ${portfolio_analysis['total_assets']:,.0f}")

        # Diversification
        div = portfolio_analysis['diversification']
        print(f"\n🌍 DIVERSIFICATION:")
        print(f"   Score: {div['diversification_score']:.0f}/100")
        print(f"   Risque de concentration: {div['concentration_risk']}")
        if div['sector_allocation']:
            print("   Allocation par secteur:")
            for sector, pct in div['sector_allocation'].items():
                print(f"   • {sector}: {pct:.1f}%")

        # Risque
        risk = portfolio_analysis['risk_metrics']
        if risk:
            print(f"\n⚠️  RISQUE:")
            print(f"   Niveau: {risk.get('risk_level', 'INCONNU')}")
            print(f"   Volatilité estimée: {risk.get('estimated_volatility', 0):.1f}%")
            print(f"   Drawdown potentiel: {risk.get('max_drawdown_potential', 0):.1f}%")
            print(f"   VaR 95%: ${risk.get('var_95', 0):,.0f}")

        print("\n" + "="*60)

    def generate_investment_recommendations(self, portfolio_analysis: Dict = None):
        """Génère des recommandations d'investissement basées sur le profil"""
        print("\n" + "="*60)
        print("💡 RECOMMANDATIONS D'INVESTISSEMENT")
        print("="*60)

        recommendations = []

        # 1. Actions recommandées pour portefeuille existant
        if portfolio_analysis and portfolio_analysis['positions']:
            print("\n📊 POUR VOTRE PORTEFEUILLE ACTUEL:")

            for symbol, analysis in portfolio_analysis['positions'].items():
                if analysis and analysis['recommendation']['action'] != 'HOLD':
                    rec = analysis['recommendation']
                    print(f"\n{symbol}:")
                    print(f"   Action: {rec['action']}")
                    print(f"   Pourcentage: {rec['percentage']}%")
                    print(f"   Raison: {rec['reason']}")

                    if rec['target_price']:
                        print(f"   Objectif: ${rec['target_price']:.2f}")
                    if rec['stop_loss']:
                        print(f"   Stop-loss: ${rec['stop_loss']:.2f}")

                    recommendations.append({
                        'type': 'PORTFOLIO_ADJUSTMENT',
                        'symbol': symbol,
                        'recommendation': rec
                    })

        # 2. Nouveaux investissements si cash disponible
        if self.user_cash > 0:
            print(f"\n💵 NOUVEAUX INVESTISSEMENTS (${self.user_cash:,.0f} disponible):")

            # Suggestions basées sur le profil
            suggestions = self.suggest_new_investments()

            for i, suggestion in enumerate(suggestions[:5], 1):  # Top 5
                print(f"\n{i}. {suggestion['symbol']} ({suggestion['type']}):")
                print(f"   Allocation suggérée: ${suggestion['suggested_investment']:,.0f}")
                print(f"   Raison: {suggestion['reason']}")
                print(f"   Risque: {suggestion['risk_level']}")

                recommendations.append({
                    'type': 'NEW_INVESTMENT',
                    'suggestion': suggestion
                })

        # 3. Marchés recommandés
        print(f"\n🌍 MARCHÉS RECOMMANDÉS:")
        market_recommendations = self.recommend_markets()

        for market in market_recommendations:
            print(f"\n📈 {market['name'].upper()}:")
            print(f"   Potentiel: {market['potential']}")
            print(f"   Horizon: {market['time_horizon']}")
            print(f"   ETFs/Symbols: {', '.join(market['etfs'][:3])}")

        return recommendations

    def suggest_new_investments(self) -> List[Dict]:
        """Suggère de nouveaux investissements basés sur le profil"""
        suggestions = []

        # Map des profils aux allocations
        profile_allocations = {
            'conservateur': {
                'bonds': 0.4,
                'dividend_stocks': 0.3,
                'gold': 0.1,
                'reits': 0.1,
                'tech': 0.1
            },
            'modéré': {
                'tech': 0.25,
                'healthcare': 0.2,
                'sp500': 0.25,
                'dividend_stocks': 0.15,
                'crypto': 0.05,
                'emerging': 0.1
            },
            'agressif': {
                'tech': 0.35,
                'crypto': 0.2,
                'biotech': 0.15,
                'emerging': 0.15,
                'small_cap': 0.15
            }
        }

        # Récupérer le profil en français
        profile_map = {
            'conservateur': 'conservateur',
            'modéré': 'modéré',
            'moderate': 'modéré',
            'agressif': 'agressif',
            'aggressive': 'agressif'
        }

        profile = profile_map.get(self.risk_profile.lower(), 'modéré')
        allocations = profile_allocations.get(profile, profile_allocations['modéré'])

        # Suggestions par catégorie
        investment_categories = {
            'tech': {
                'symbols': ['QQQ', 'VGT', 'MSFT', 'AAPL', 'NVDA'],
                'reason': 'Croissance technologique long terme',
                'risk': 'Élevé'
            },
            'dividend_stocks': {
                'symbols': ['VYM', 'SCHD', 'JNJ', 'PG', 'XOM'],
                'reason': 'Revenus réguliers + stabilité',
                'risk': 'Faible-Modéré'
            },
            'sp500': {
                'symbols': ['SPY', 'VOO', 'IVV'],
                'reason': 'Exposition large marché US',
                'risk': 'Modéré'
            },
            'crypto': {
                'symbols': ['GBTC', 'ETHE', 'BTC-USD', 'ETH-USD'],
                'reason': 'Potentiel haute croissance',
                'risk': 'Très élevé'
            },
            'healthcare': {
                'symbols': ['XLV', 'VHT', 'UNH', 'JNJ'],
                'reason': 'Secteur défensif + innovation',
                'risk': 'Modéré'
            },
            'bonds': {
                'symbols': ['BND', 'AGG', 'TLT'],
                'reason': 'Stabilité capital',
                'risk': 'Faible'
            },
            'gold': {
                'symbols': ['GLD', 'IAU'],
                'reason': 'Hedge contre inflation',
                'risk': 'Faible-Modéré'
            },
            'reits': {
                'symbols': ['VNQ', 'O', 'AMT'],
                'reason': 'Revenus immobiliers',
                'risk': 'Modéré'
            },
            'emerging': {
                'symbols': ['VWO', 'EEM', 'SCHE'],
                'reason': 'Croissance marchés émergents',
                'risk': 'Élevé'
            },
            'biotech': {
                'symbols': ['XBI', 'IBB', 'REGN'],
                'reason': 'Potentiel innovation médicale',
                'risk': 'Très élevé'
            },
            'small_cap': {
                'symbols': ['IJR', 'VB', 'IWM'],
                'reason': 'Potentiel croissance',
                'risk': 'Élevé'
            }
        }

        # Filtrer selon préférences/exclusions
        filtered_categories = {}
        for cat, info in investment_categories.items():
            # Vérifier exclusions
            exclude = False
            for excluded in self.excluded_sectors:
                if excluded in cat:
                    exclude = True
                    break

            if not exclude:
                # Vérifier préférences
                if not self.market_preferences:
                    filtered_categories[cat] = info
                else:
                    # Vérifier si une préférence correspond à cette catégorie
                    category_tags = {
                        'tech': ['technologie', 'tech', 'informatique'],
                        'healthcare': ['santé', 'médical', 'pharma'],
                        'crypto': ['crypto', 'bitcoin', 'ethereum'],
                        'dividend_stocks': ['dividendes', 'revenus'],
                        'bonds': ['obligations', 'titres', 'fixed income'],
                        'gold': ['or', 'métaux', 'précieux'],
                        'reits': ['immobilier', 'reit', 'property'],
                        'emerging': ['émergents', 'emerging', 'pays en développement'],
                        'biotech': ['biotech', 'biotechnologie', 'médical'],
                        'small_cap': ['small cap', 'petites entreprises']
                    }

                    tags = category_tags.get(cat, [])
                    if any(pref in tag for pref in self.market_preferences for tag in tags):
                        filtered_categories[cat] = info

        # Générer suggestions
        total_investment = self.user_cash

        for category, allocation_pct in allocations.items():
            if category in filtered_categories and allocation_pct > 0:
                suggested_investment = total_investment * allocation_pct

                if suggested_investment > 100:  # Seuil minimum
                    category_info = filtered_categories[category]

                    for symbol in category_info['symbols'][:2]:  # 2 premiers symboles
                        suggestions.append({
                            'symbol': symbol,
                            'type': category.upper(),
                            'suggested_investment': suggested_investment / 2,  # Split entre symboles
                            'reason': category_info['reason'],
                            'risk_level': category_info['risk'],
                            'allocation_pct': allocation_pct * 100
                        })

        # Trier par allocation
        suggestions.sort(key=lambda x: x['suggested_investment'], reverse=True)
        return suggestions

    def recommend_markets(self) -> List[Dict]:
        """Recommande des marchés basés sur conditions actuelles"""
        markets = [
            {
                'name': 'Technologie US',
                'potential': 'Élevé',
                'time_horizon': '6-18 mois',
                'reason': 'Innovation IA, cloud computing',
                'etfs': ['QQQ', 'VGT', 'XLK'],
                'risk': 'Élevé'
            },
            {
                'name': 'Énergie Renouvelable',
                'potential': 'Moyen-Élevé',
                'time_horizon': '2-5 ans',
                'reason': 'Transition énergétique mondiale',
                'etfs': ['ICLN', 'TAN', 'PBW'],
                'risk': 'Moyen-Élevé'
            },
            {
                'name': 'Marchés Émergents',
                'potential': 'Élevé',
                'time_horizon': '3-7 ans',
                'reason': 'Croissance démographique, urbanisation',
                'etfs': ['VWO', 'EEM', 'IEMG'],
                'risk': 'Élevé'
            },
            {
                'name': 'Santé & Biotech',
                'potential': 'Moyen',
                'time_horizon': '2-4 ans',
                'reason': 'Vieillissement population, innovation médicale',
                'etfs': ['XLV', 'IBB', 'VHT'],
                'risk': 'Moyen'
            },
            {
                'name': 'Cryptomonnaies',
                'potential': 'Très élevé',
                'time_horizon': '1-3 ans',
                'reason': 'Adoption institutionnelle, halving cycles',
                'etfs': ['GBTC', 'ETHE', 'BITO'],
                'risk': 'Très élevé'
            },
            {
                'name': 'Dividendes US',
                'potential': 'Stable',
                'time_horizon': 'Long terme',
                'reason': 'Revenus réguliers, entreprises matures',
                'etfs': ['VYM', 'SCHD', 'DGRO'],
                'risk': 'Faible-Modéré'
            }
        ]

        # Filtrer selon préférences
        if self.market_preferences:
            filtered_markets = []
            for market in markets:
                market_lower = market['name'].lower()
                if any(pref in market_lower for pref in self.market_preferences):
                    filtered_markets.append(market)
            return filtered_markets[:3]  # Top 3

        # Retourner tous si pas de préférences
        return markets[:4]

    def run_complete_analysis(self, fast_mode: bool = True):
        """Exécute l'analyse complète"""
        print("\n" + "="*60)
        print("🤖 AGENT DE TRADING - ANALYSE COMPLÈTE")
        if fast_mode:
            print("⚡ MODE RAPIDE ACTIVÉ")
        print("="*60)

        # 1. Configuration utilisateur
        self.setup_user_profile()

        # 2. Portefeuille existant
        if input("\nAvez-vous des positions actuelles? (oui/non) [non]: ").strip().lower().startswith('o'):
            self.input_current_portfolio()

        # 3. Analyse
        print("\n⏳ Analyse en cours...")

        portfolio_analysis = None
        if self.user_portfolio:
            if fast_mode:
                portfolio_analysis = self.analyze_user_portfolio_fast()
            else:
                portfolio_analysis = self.analyze_user_portfolio()
        else:
            print("Aucune position à analyser.")

        # 4. Recommandations
        recommendations = self.generate_investment_recommendations(portfolio_analysis)

        # 5. Plan d'action
        self.generate_action_plan(portfolio_analysis, recommendations)

        return {
            'profile': {
                'risk': self.risk_profile,
                'cash': self.user_cash,
                'preferences': self.market_preferences,
                'exclusions': self.excluded_sectors
            },
            'portfolio': portfolio_analysis,
            'recommendations': recommendations
        }

    def generate_action_plan(self, portfolio_analysis, recommendations):
        """Génère un plan d'action concret"""
        print("\n" + "="*60)
        print("🎯 PLAN D'ACTION CONCRET")
        print("="*60)

        print("\n📋 ACTIONS IMMÉDIATES:")

        action_counter = 1

        # Actions pour portefeuille existant
        if portfolio_analysis and portfolio_analysis['positions']:
            for symbol, analysis in portfolio_analysis['positions'].items():
                if analysis and analysis['recommendation']['action'] != 'HOLD':
                    rec = analysis['recommendation']
                    shares_to_action = analysis['shares'] * (rec['percentage'] / 100)

                    print(f"\n{action_counter}. {symbol}:")
                    if rec['action'] in ['TAKE_PROFIT', 'CUT_LOSSES']:
                        print(f"   → Vendre {rec['percentage']}% ({shares_to_action:.0f} actions)")
                        print(f"   → Prix cible: ${rec.get('target_price', analysis['current_price']):.2f}")
                    elif rec['action'] in ['AVERAGE_DOWN', 'ADD_TO_WINNERS']:
                        print(f"   → Acheter {rec['percentage']}% supplémentaire")
                        print(f"   → Montant estimé: ${shares_to_action * analysis['current_price']:,.0f}")

                    action_counter += 1

        # Nouvelles opportunités
        if self.user_cash > 0:
            print(f"\n💰 NOUVELLES OPPORTUNITÉS (${self.user_cash:,.0f}):")

            new_investments = [r for r in recommendations if r['type'] == 'NEW_INVESTMENT']
            if new_investments:
                for rec in new_investments[:3]:  # Limiter à 3
                    suggestion = rec['suggestion']
                    print(f"\n{action_counter}. {suggestion['symbol']}:")
                    print(f"   → Investir: ${suggestion['suggested_investment']:,.0f}")
                    print(f"   → Catégorie: {suggestion['type']}")
                    print(f"   → Raison: {suggestion['reason']}")

                    action_counter += 1
            else:
                print("   → Aucune nouvelle opportunité identifiée")

        # Allocation cash restante
        total_planned_investment = sum(
            rec['suggestion']['suggested_investment']
            for rec in recommendations
            if rec['type'] == 'NEW_INVESTMENT'
        )

        cash_remaining = self.user_cash - total_planned_investment

        if cash_remaining > 0:
            print(f"\n💵 CASH RESTANT: ${cash_remaining:,.0f}")
            print("   → Garder en réserve pour opportunités futures")
            print("   → OU Investir en fonds monétaires (BIL, SHV)")

        print("\n" + "="*60)
        print("⏰ PROCHAINES ÉTAPES:")
        print("1. Exécuter les transactions recommandées")
        print("2. Revoir dans 1-2 semaines")
        print("3. Ajuster stop-loss si nécessaire")
        print("="*60)

# Interface principale
def main():
    """Point d'entrée principal - OPTIMISÉ"""
    print("\n" + "="*60)
    print("🤖 AGENT DE TRADING PERSONNALISÉ - VERSION OPTIMISÉE")
    print("="*60)
    print("Analyser votre portefeuille + recommandations d'investissement\n")

    # Mode rapide par défaut
    print("⚡ Le mode rapide est activé par défaut pour une analyse plus rapide.")
    print("   (Désactivez-le dans les options pour une analyse approfondie)\n")

    # Initialisation
    agent = PortfolioTradingAgent()

    try:
        # Demander le mode
        config = input("Configurer les options? (oui/non) [non]: ").strip().lower()

        fast_mode = True
        if config.startswith('o'):
            fast_choice = input("Mode rapide? (oui/non) [oui]: ").strip().lower()
            fast_mode = not fast_choice.startswith('n')

        # Exécution avec chronométrage
        start_time = datetime.now()
        results = agent.run_complete_analysis(fast_mode=fast_mode)
        end_time = datetime.now()

        # Afficher le temps d'exécution
        duration = (end_time - start_time).seconds
        print(f"\n⏱️  Temps d'analyse total: {duration} secondes")

        # Sauvegarde des résultats
        save_option = input("\n📁 Sauvegarder les résultats? (oui/non) [non]: ").strip().lower()
        if save_option.startswith('o'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_report_{timestamp}.json"

            # Conversion pour JSON
            json_results = {
                'timestamp': timestamp,
                'analysis_duration_seconds': duration,
                'fast_mode': fast_mode,
                'profile': results['profile'],
                'portfolio_summary': {
                    'total_value': results['portfolio']['total_value'] if results['portfolio'] else 0,
                    'performance': results['portfolio']['total_performance_pct'] if results['portfolio'] else 0
                } if results['portfolio'] else None,
                'recommendations_count': len(results['recommendations'])
            }

            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2)
            print(f"✓ Rapport sauvegardé: {filename}")

        # Recommencer?
        restart = input("\n🔄 Effectuer une nouvelle analyse? (oui/non) [non]: ").strip().lower()
        if restart.startswith('o'):
            main()
        else:
            print("\n✅ Analyse terminée. Bon investissement!")

    except KeyboardInterrupt:
        print("\n\n❌ Analyse interrompue par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        print("Veuillez réessayer")

if __name__ == "__main__":
    main()