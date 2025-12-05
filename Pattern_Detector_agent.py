import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks, argrelextrema
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

class PatternEncoder:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.pattern_codes = {
            'NO_PATTERN': [0, 0, 0],
            'HEAD_SHOULDERS': [1, 0, 0],
            'INVERSE_HEAD_SHOULDERS': [0, 1, 0],
            'DOUBLE_TOP': [0, 0, 1],
            'DOUBLE_BOTTOM': [1, 1, 0],
            'TRIANGLE_ASCENDING': [1, 0, 1],
            'TRIANGLE_DESCENDING': [0, 1, 1],
            'TRIANGLE_SYMMETRICAL': [1, 1, 1],
            'FLAG_BULLISH': [1, 0, 0, 1],
            'FLAG_BEARISH': [0, 1, 0, 1]
        }
    
    def encode_pattern_features(self, price_series):
        features = []
        
        features.append(np.mean(price_series))
        features.append(np.std(price_series))
        if price_series[0] != 0:
            features.append((price_series[-1] - price_series[0]) / price_series[0])
        else:
            features.append(0)
        
        if np.mean(price_series) != 0:
            features.append(np.ptp(price_series) / np.mean(price_series))
        else:
            features.append(0)
        
        x = np.arange(len(price_series))
        slope, intercept = np.polyfit(x, price_series, 1)
        features.append(slope)
        features.append(intercept)
        
        quad_coeff = np.polyfit(x, price_series, 2)[0]
        features.append(quad_coeff)
        
        peaks_idx = argrelextrema(price_series, np.greater, order=3)[0]
        troughs_idx = argrelextrema(price_series, np.less, order=3)[0]
        
        features.append(len(peaks_idx))
        features.append(len(troughs_idx))
        
        if len(peaks_idx) > 0:
            peaks = price_series[peaks_idx]
            features.append(np.mean(peaks))
            features.append(np.std(peaks))
            if len(peaks) > 1:
                features.append((peaks[-1] - peaks[0]) / len(peaks))
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0])
        
        if len(troughs_idx) > 0:
            troughs = price_series[troughs_idx]
            features.append(np.mean(troughs))
            features.append(np.std(troughs))
            if len(troughs) > 1:
                features.append((troughs[-1] - troughs[0]) / len(troughs))
            else:
                features.append(0)
        else:
            features.extend([0, 0, 0])
        
        price_range = np.max(price_series) - np.min(price_series)
        features.append(price_range)
        
        if np.mean(price_series) != 0:
            features.append((np.max(price_series) - np.mean(price_series)) / np.mean(price_series))
            features.append((np.mean(price_series) - np.min(price_series)) / np.mean(price_series))
        else:
            features.extend([0, 0])
        
        if len(price_series) > 1:
            returns = np.diff(price_series) / price_series[:-1]
            features.append(np.std(returns) if len(returns) > 0 else 0)
            features.append(np.mean(np.abs(returns)) if len(returns) > 0 else 0)
        else:
            features.extend([0, 0])
        
        if len(price_series) > 1:
            corr = np.corrcoef(price_series[:-1], price_series[1:])[0, 1]
            features.append(corr if not np.isnan(corr) else 0)
        else:
            features.append(0)
        
        ma_short = np.mean(price_series[-5:] if len(price_series) >= 5 else price_series)
        ma_long = np.mean(price_series)
        if ma_long != 0:
            features.append(ma_short / ma_long)
        else:
            features.append(1)
        
        if price_range != 0:
            features.append((price_series[-1] - np.min(price_series)) / price_range)
        else:
            features.append(0.5)
        
        if len(features) < 30:
            features.extend([0] * (30 - len(features)))
        elif len(features) > 30:
            features = features[:30]
        
        return np.array(features)
    
    def get_pattern_label_vector(self, pattern_name):
        if pattern_name in self.pattern_codes:
            code = self.pattern_codes[pattern_name]
            if len(code) < 10:
                code = code + [0] * (10 - len(code))
            return np.array(code)
        else:
            return np.zeros(10)

class PatternLibrary:
    def __init__(self):
        self.encoder = PatternEncoder()
        self.pattern_samples = []
    
    def add_pattern_sample(self, price_data, pattern_name):
        features = self.encoder.encode_pattern_features(price_data)
        label_vector = self.encoder.get_pattern_label_vector(pattern_name)
        self.pattern_samples.append((features, label_vector))
        return features
    
    def create_synthetic_patterns(self):
        print("Creating synthetic pattern training data...")
        
        patterns_to_create = [
            ('HEAD_SHOULDERS', self._create_head_shoulders),
            ('INVERSE_HEAD_SHOULDERS', self._create_inverse_head_shoulders),
            ('DOUBLE_TOP', self._create_double_top),
            ('DOUBLE_BOTTOM', self._create_double_bottom),
            ('TRIANGLE_ASCENDING', self._create_triangle_ascending),
            ('TRIANGLE_DESCENDING', self._create_triangle_descending),
            ('TRIANGLE_SYMMETRICAL', self._create_triangle_symmetrical),
            ('FLAG_BULLISH', self._create_flag_bullish),
            ('FLAG_BEARISH', self._create_flag_bearish),
            ('NO_PATTERN', self._create_no_pattern)
        ]
        
        for pattern_name, create_func in patterns_to_create:
            for _ in range(10):
                price_data = create_func()
                self.add_pattern_sample(price_data, pattern_name)
        
        print(f"Created {len(self.pattern_samples)} pattern samples")
        return self
    
    def _create_head_shoulders(self):
        x = np.linspace(0, 4*np.pi, 30)
        return 100 + 10*np.sin(x) + 5*np.sin(2*x) + np.random.randn(30) * 2
    
    def _create_inverse_head_shoulders(self):
        x = np.linspace(0, 4*np.pi, 30)
        return 100 - 10*np.sin(x) - 5*np.sin(2*x) + np.random.randn(30) * 2
    
    def _create_double_top(self):
        x = np.linspace(0, 3*np.pi, 30)
        return 100 + 8*np.abs(np.sin(x)) + np.random.randn(30) * 3
    
    def _create_double_bottom(self):
        x = np.linspace(0, 3*np.pi, 30)
        return 100 - 8*np.abs(np.sin(x)) + np.random.randn(30) * 3
    
    def _create_triangle_ascending(self):
        x = np.linspace(0, 3*np.pi, 30)
        return 95 + np.linspace(0, 10, 30) + 5*np.sin(x) + np.random.randn(30) * 2
    
    def _create_triangle_descending(self):
        x = np.linspace(0, 3*np.pi, 30)
        return 105 - np.linspace(0, 10, 30) + 5*np.sin(x) + np.random.randn(30) * 2
    
    def _create_triangle_symmetrical(self):
        x = np.linspace(0, 3*np.pi, 30)
        amplitude = 15 - np.linspace(0, 14, 30)
        return 100 + amplitude * np.sin(x) + np.random.randn(30) * 2
    
    def _create_flag_bullish(self):
        x = np.linspace(0, 3*np.pi, 30)
        return 90 + np.linspace(0, 15, 30) + 3*np.sin(x*2) + np.random.randn(30) * 2
    
    def _create_flag_bearish(self):
        x = np.linspace(0, 3*np.pi, 30)
        return 110 - np.linspace(0, 15, 30) + 3*np.sin(x*2) + np.random.randn(30) * 2
    
    def _create_no_pattern(self):
        returns = np.random.randn(30) * 0.5
        return 100 + np.cumsum(returns)
    
    def get_training_data(self):
        X = []
        y = []
        
        for features, label_vector in self.pattern_samples:
            X.append(features)
            y.append(label_vector)
        
        return np.array(X), np.array(y)

class PatternRecognitionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {
            'patterns': None,
            'encoder': None
        }
        self.is_trained = False
    
    def train_from_library(self, pattern_library):
        print("Training pattern recognition model...")
        
        X, y = pattern_library.get_training_data()
        
        if len(X) == 0:
            raise ValueError("No training data available!")
        
        X_scaled = self.scaler.fit_transform(X)
        
        from sklearn.multioutput import MultiOutputClassifier
        
        base_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            class_weight='balanced'
        )
        
        self.models['patterns'] = MultiOutputClassifier(base_model)
        self.models['patterns'].fit(X_scaled, y)
        self.models['encoder'] = pattern_library.encoder
        self.is_trained = True
        
        train_predictions = self.models['patterns'].predict(X_scaled)
        train_accuracy = np.mean(train_predictions == y)
        print(f"Training accuracy: {train_accuracy:.2%}")
        
        return self
    
    def predict_pattern(self, price_window):
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        features = self.models['encoder'].encode_pattern_features(price_window)
        features_scaled = self.scaler.transform([features])
        
        prediction_vector = self.models['patterns'].predict(features_scaled)[0]
        
        pattern_name = self._decode_prediction(prediction_vector)
        
        confidence = 0.8
        try:
            probas = self.models['patterns'].predict_proba(features_scaled)
            if probas and len(probas) > 0:
                confidence = np.mean([prob[0][1] for prob in probas if prob.shape[1] > 1])
        except:
            pass
        
        return pattern_name, confidence, prediction_vector
    
    def _decode_prediction(self, prediction_vector):
        min_distance = float('inf')
        best_pattern = 'NO_PATTERN'
        
        for pattern_name, pattern_code in self.models['encoder'].pattern_codes.items():
            code_padded = pattern_code + [0] * (len(prediction_vector) - len(pattern_code))
            distance = np.sum(np.abs(np.array(code_padded) - prediction_vector))
            
            if distance < min_distance:
                min_distance = distance
                best_pattern = pattern_name
        
        return best_pattern
    
    def visualize_prediction(self, price_window, prediction_result):
        pattern_name, confidence, pred_vector = prediction_result
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(price_window, 'b-', linewidth=2, label='Price')
        axes[0].fill_between(range(len(price_window)), 
                           np.min(price_window), 
                           price_window, 
                           alpha=0.3)
        axes[0].set_title(f"Detected Pattern: {pattern_name} (Confidence: {confidence:.2%})")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Price")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].bar(range(len(pred_vector)), pred_vector, alpha=0.7)
        axes[1].set_title("Pattern Encoding Vector (ML Representation)")
        axes[1].set_xlabel("Feature Dimension")
        axes[1].set_ylabel("Activation")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

class RealTimePatternScanner:
    def __init__(self, trained_model, window_size=30):
        self.model = trained_model
        self.window_size = window_size
        self.price_buffer = []
        
    def add_price(self, price):
        self.price_buffer.append(price)
        
        if len(self.price_buffer) > self.window_size * 2:
            self.price_buffer = self.price_buffer[-self.window_size * 2:]
        
        if len(self.price_buffer) >= self.window_size:
            current_window = self.price_buffer[-self.window_size:]
            return self.scan_pattern(current_window)
        
        return None
    
    def scan_pattern(self, price_window):
        try:
            pattern_result = self.model.predict_pattern(np.array(price_window))
            pattern_name, confidence, _ = pattern_result
            
            if pattern_name != 'NO_PATTERN' and confidence > 0.6:
                return {
                    'pattern': pattern_name,
                    'confidence': confidence,
                    'window': price_window.copy(),
                    'timestamp': datetime.now()
                }
        except Exception as e:
            print(f"Error scanning pattern: {e}")
        
        return None

def main():
    print("=" * 60)
    print("PATTERN RECOGNITION ML SYSTEM")
    print("=" * 60)
    
    print("\n1. Creating pattern library...")
    library = PatternLibrary()
    library.create_synthetic_patterns()
    
    print("\n2. Training ML model...")
    model = PatternRecognitionModel()
    model.train_from_library(library)
    
    print("\n3. Testing with generated patterns...")
    
    test_patterns = [
        ('HEAD_SHOULDERS', library._create_head_shoulders()),
        ('DOUBLE_TOP', library._create_double_top()),
        ('TRIANGLE_ASCENDING', library._create_triangle_ascending()),
        ('FLAG_BULLISH', library._create_flag_bullish()),
        ('NO_PATTERN', library._create_no_pattern())
    ]
    
    for pattern_name, test_data in test_patterns:
        result = model.predict_pattern(test_data)
        print(f"  {pattern_name}: Detected as {result[0]} (confidence: {result[1]:.2%})")
    
    print("\n4. Real-time pattern detection simulation...")
    print("-" * 50)
    
    scanner = RealTimePatternScanner(model)
    
    prices = []
    detected_count = 0
    
    for i in range(200):
        if i == 50:
            hs_pattern = library._create_head_shoulders()
            for price in hs_pattern:
                result = scanner.add_price(price)
                prices.append(price)
                if result and result['pattern'] != 'NO_PATTERN':
                    print(f"[Step {i}] Detected: {result['pattern']} "
                          f"(confidence: {result['confidence']:.2%})")
                    detected_count += 1
        elif i == 120:
            dt_pattern = library._create_double_top()
            for price in dt_pattern:
                result = scanner.add_price(price)
                prices.append(price)
                if result and result['pattern'] != 'NO_PATTERN':
                    print(f"[Step {i}] Detected: {result['pattern']} "
                          f"(confidence: {result['confidence']:.2%})")
                    detected_count += 1
        else:
            if i == 0:
                price = 100 + np.random.randn() * 0.5
            else:
                price = prices[-1] + np.random.randn() * 0.5
            
            result = scanner.add_price(price)
            prices.append(price)
            
            if result and result['pattern'] != 'NO_PATTERN':
                print(f"[Step {i}] Detected: {result['pattern']} "
                      f"(confidence: {result['confidence']:.2%})")
                detected_count += 1
    
    print(f"\nTotal patterns detected during simulation: {detected_count}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(prices, 'b-', alpha=0.7, linewidth=1)
    plt.title("Price Stream with Pattern Detection")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.grid(True, alpha=0.3)
    
    if detected_count > 0:
        plt.axvspan(50, 80, alpha=0.2, color='yellow', label='Pattern Area 1')
        plt.axvspan(120, 150, alpha=0.2, color='orange', label='Pattern Area 2')
        plt.legend()
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("System ready for real-time pattern detection!")
    print("=" * 60)
    
    return model, scanner

if __name__ == "__main__":
    trained_model, real_time_scanner = main()
    
    def test_custom_pattern():
        print("\n" + "=" * 60)
        print("CUSTOM PATTERN TEST")
        print("=" * 60)
        
        x = np.linspace(0, 4*np.pi, 30)
        custom_pattern = 100 + 12*np.sin(x) + 6*np.sin(2*x) + np.random.randn(30) * 1.5
        
        pattern_name, confidence, pred_vector = trained_model.predict_pattern(custom_pattern)
        
        print(f"\nPattern detected: {pattern_name}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Prediction vector: {pred_vector}")
        
        trained_model.visualize_prediction(custom_pattern, (pattern_name, confidence, pred_vector))
        
        return pattern_name, confidence
    
    test_result = test_custom_pattern()
    print(f"\nTest completed successfully!")