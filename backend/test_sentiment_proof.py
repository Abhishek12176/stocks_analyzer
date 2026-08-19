import sys, json
sys.path.insert(0, '.')
from app.services.signal_service import generate_trade_signal

# Scenario: All technicals say BUY but news is terrible
sig = generate_trade_signal(
    price=100.0, rsi=25.0, macd=2.5, signal=1.0, sma20=105.0, sma50=100.0,
    sentiment_score=-0.70,
)
print(\"=== TECHNICALS SAY BUY, NEWS SAYS NEGATIVE ===\")
print(json.dumps(sig, indent=2))
