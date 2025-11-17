"""
Price Direction Prediction using CNN and Chart Images
Author: [sklepfuzja]
Date: 2024

Description: CNN-based price direction prediction using OHLC chart images with technical indicators.
Uses AlexNet architecture for image classification of bullish/bearish patterns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from datetime import datetime

# ML & Deep Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam

# Data & Trading Imports
import MetaTrader5 as mt5
import pandas_ta as ta
from PIL import Image

# Configuration
warnings.filterwarnings('ignore', category=RuntimeWarning)
tf.config.set_soft_device_placement(True)

# ==================== CONFIGURATION ====================
class Config:
    """Configuration parameters for the CNN price prediction system."""
    
    # MT5 Settings
    SYMBOL = 'EURUSD'
    TIMEFRAME = mt5.TIMEFRAME_H1
    MT5_LOGIN = None
    MT5_PASSWORD = None
    MT5_SERVER = None
    
    # Data Settings
    DATA_POINTS = 3000
    WINDOW_SIZE = 120  # Number of candles per image
    IMAGE_SIZE = (128, 128)
    
    # Model Settings
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Paths
    IMAGE_SAVE_DIR = 'ohlc_images'
    
    # Technical Indicators
    EMA_PERIODS = [5, 10, 15]
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2

# ==================== DATA FETCHER ====================
class DataFetcher:
    """Handles data fetching from MT5."""
    
    def __init__(self, login, password, server):
        """Initialize MT5 connection."""
        self.fetcher = self._initialize_mt5(login, password, server)
    
    def _initialize_mt5(self, login, password, server):
        """Initialize MT5 connection."""
        # Placeholder for your MT5 initialization
        # This should be replaced with your actual DataFetcherMT5 implementation
        class MockFetcher:
            def fetch_data_candle(self, symbol, timeframe, points):
                # Mock implementation - replace with actual MT5 data fetching
                dates = pd.date_range(start='2024-01-01', periods=points, freq='H')
                data = {
                    'open': np.random.uniform(1.05, 1.10, points).cumsum(),
                    'high': np.random.uniform(1.06, 1.11, points).cumsum(),
                    'low': np.random.uniform(1.04, 1.09, points).cumsum(),
                    'close': np.random.uniform(1.05, 1.10, points).cumsum()
                }
                return pd.DataFrame(data, index=dates)
        
        return MockFetcher()
    
    def fetch_data(self, symbol, timeframe, points):
        """Fetch OHLC data from MT5."""
        df = self.fetcher.fetch_data_candle(symbol, timeframe, points)
        df.rename(columns={
            'open': 'Open', 'high': 'High', 
            'low': 'Low', 'close': 'Close'
        }, inplace=True)
        return df

# ==================== IMAGE GENERATOR ====================
class OHLCImageGenerator:
    """Generates OHLC chart images with technical indicators."""
    
    def __init__(self, config):
        self.config = config
    
    def calculate_indicators(self, df):
        """Calculate technical indicators for the dataset."""
        # Exponential Moving Averages
        for period in self.config.EMA_PERIODS:
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        
        # Bollinger Bands
        df['SMA_20'] = df['Close'].rolling(window=self.config.BOLLINGER_PERIOD).mean()
        df['BB_upper'] = df['SMA_20'] + (
            df['Close'].rolling(window=self.config.BOLLINGER_PERIOD).std() * self.config.BOLLINGER_STD
        )
        df['BB_lower'] = df['SMA_20'] - (
            df['Close'].rolling(window=self.config.BOLLINGER_PERIOD).std() * self.config.BOLLINGER_STD
        )
        
        return df
    
    def generate_images(self, df, save_dir):
        """
        Generate OHLC chart images with technical indicators.
        
        Args:
            df: DataFrame with OHLC data and indicators
            save_dir: Directory to save generated images
            
        Returns:
            labels: List of binary labels (1=bullish, 0=bearish)
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        labels = []
        window_size = self.config.WINDOW_SIZE
        
        for i in range(len(df) - window_size):
            # Extract window data
            ohlc_window = df.iloc[i:i + window_size]
            
            # Create label based on next candle movement
            current_close = df['Close'].iloc[i + window_size - 1]
            next_close = df['Close'].iloc[i + window_size]
            label = 1 if next_close > current_close else 0
            labels.append(label)
            
            # Generate and save chart image
            self._create_chart_image(ohlc_window, save_dir, i)
        
        return labels
    
    def _create_chart_image(self, ohlc_window, save_dir, index):
        """Create and save a single chart image."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot OHLC lines
        ax.plot(ohlc_window.index, ohlc_window['Open'], label='Open', color='blue', linewidth=1)
        ax.plot(ohlc_window.index, ohlc_window['High'], label='High', color='green', linewidth=1)
        ax.plot(ohlc_window.index, ohlc_window['Low'], label='Low', color='red', linewidth=1)
        ax.plot(ohlc_window.index, ohlc_window['Close'], label='Close', color='black', linewidth=1)
        
        # Plot EMAs
        colors = ['purple', 'orange', 'brown']
        for i, period in enumerate(self.config.EMA_PERIODS):
            ax.plot(ohlc_window.index, ohlc_window[f'EMA_{period}'], 
                   label=f'EMA {period}', color=colors[i], linestyle='--')
        
        # Plot Bollinger Bands
        ax.plot(ohlc_window.index, ohlc_window['BB_upper'], 
               label='Bollinger Upper', color='grey', linestyle=':')
        ax.plot(ohlc_window.index, ohlc_window['BB_lower'], 
               label='Bollinger Lower', color='grey', linestyle=':')
        
        # Remove axes for pure image data
        ax.axis('off')
        
        # Save image
        image_path = os.path.join(save_dir, f'ohlc_{index:04d}.png')
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()

# ==================== CNN MODEL ====================
class PricePredictionCNN:
    """CNN model for price direction prediction."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
    
    def create_alexnet(self, input_shape):
        """Create AlexNet-inspired CNN architecture."""
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(96, (11, 11), activation='relu', 
                         input_shape=input_shape, strides=(4, 4)),
            layers.MaxPooling2D((3, 3), strides=(2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(256, (5, 5), activation='relu', padding='same'),
            layers.MaxPooling2D((3, 3), strides=(2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((3, 3), strides=(2, 2)),
            
            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')  # 2 classes: bullish/bearish
        ])
        
        return model
    
    def compile_model(self):
        """Compile the CNN model."""
        input_shape = (self.config.IMAGE_SIZE[0], self.config.IMAGE_SIZE[1], 3)
        self.model = self.create_alexnet(input_shape)
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train the CNN model."""
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1
        )
        
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=self.config.BATCH_SIZE),
            epochs=self.config.EPOCHS,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        return history

# ==================== DATA LOADER ====================
class DataLoader:
    """Handles loading and preprocessing of image data."""
    
    def __init__(self, config):
        self.config = config
    
    def load_images(self, image_dir, labels):
        """
        Load images and corresponding labels.
        
        Args:
            image_dir: Directory containing images
            labels: List of labels corresponding to images
            
        Returns:
            X: Normalized image array
            y: One-hot encoded labels
        """
        X, y = [], []
        
        # Sort files to maintain order
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        for i, filename in enumerate(image_files):
            if i >= len(labels):  # Safety check
                break
                
            img_path = os.path.join(image_dir, filename)
            image = Image.open(img_path).resize(self.config.IMAGE_SIZE)
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            
            # Ensure 3 channels (remove alpha channel if present)
            if image.shape[-1] == 4:
                image = image[..., :3]
            
            X.append(image)
            y.append(labels[i])
        
        X = np.array(X)
        y = to_categorical(y, num_classes=2)
        
        return X, y

# ==================== MAIN PIPELINE ====================
def main():
    """Main execution pipeline."""
    print("🚀 Starting CNN Price Direction Prediction Pipeline...")
    
    # Initialize configuration
    config = Config()
    
    # Setup GPU
    setup_gpu()
    
    # Fetch data
    print("📊 Fetching market data...")
    data_fetcher = DataFetcher(config.MT5_LOGIN, config.MT5_PASSWORD, config.MT5_SERVER)
    df = data_fetcher.fetch_data(config.SYMBOL, config.TIMEFRAME, config.DATA_POINTS)
    
    # Generate images
    print("🖼️ Generating OHLC chart images...")
    image_generator = OHLCImageGenerator(config)
    df_with_indicators = image_generator.calculate_indicators(df)
    labels = image_generator.generate_images(df_with_indicators, config.IMAGE_SAVE_DIR)
    
    # Load images
    print("📥 Loading images for training...")
    data_loader = DataLoader(config)
    X, y = data_loader.load_images(config.IMAGE_SAVE_DIR, labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.VALIDATION_SPLIT, random_state=42, shuffle=True
    )
    
    print(f"📈 Dataset: {X.shape[0]} images, Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    
    # Build and train model
    print("🤖 Building CNN model...")
    cnn_model = PricePredictionCNN(config)
    cnn_model.compile_model()
    
    print("🎯 Starting model training...")
    history = cnn_model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    print("📊 Evaluating model performance...")
    test_loss, test_accuracy = cnn_model.model.evaluate(X_test, y_test)
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # Generate predictions
    y_pred = cnn_model.model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true_classes, y_pred_classes, 
                              target_names=['Bearish', 'Bullish']))
    
    # Plot training history
    plot_training_history(history)
    
    print("🎉 Pipeline completed successfully!")

# ==================== UTILITY FUNCTIONS ====================
def setup_gpu():
    """Configure GPU settings."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s)")
        try:
            # Limit GPU memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU found, using CPU")

def plot_training_history(history):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()