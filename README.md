# Price Direction Prediction using CNN and Chart Images

Deep learning system for financial market direction prediction using convolutional neural networks and OHLC chart images.

## Features

- **OHLC Chart Image Generation**: Converts price data to visual charts with technical indicators
- **AlexNet CNN Architecture**: Deep learning model for image classification
- **Technical Indicators**: EMA (5,10,15) and Bollinger Bands on generated charts
- **Bullish/Bearish Classification**: Binary prediction of next candle direction
- **GPU Acceleration**: Optimized for TensorFlow GPU execution

## Technical Architecture
Data Fetching → Indicator Calculation → Image Generation → CNN Training → Prediction

## Model Details

- **Input**: 128x128 pixel OHLC chart images with technical indicators
- **Architecture**: AlexNet-inspired CNN with 5 convolutional layers
- **Output**: Binary classification (Bullish/Bearish)
- **Training**: 20 epochs with data augmentation

# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Disclaimer
This trading system is for educational and research purposes. Always test strategies thoroughly with historical data and paper trading before deploying with real capital. Past performance does not guarantee future results.
