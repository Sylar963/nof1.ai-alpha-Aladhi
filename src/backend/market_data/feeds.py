from dataclasses import dataclass
from urllib.parse import urlencode


SOURCE_NAMES = ("binance_spot", "binance_perp_book", "binance_perp_market", "paradex")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    url: str
    market: str
    channels: tuple[str, ...]
    subscribe_messages: tuple[dict, ...] = ()
    depth_url: str | None = None
    futures: bool = False


def build_sources(config) -> list[SourceSpec]:
    symbol = config.symbol.lower()
    spot = (f"{symbol}@bookTicker", f"{symbol}@aggTrade", f"{symbol}@depth@100ms")
    perp_book = (f"{symbol}@bookTicker", f"{symbol}@depth@100ms")
    perp_market = (f"{symbol}@aggTrade", f"{symbol}@markPrice@1s", f"{symbol}@forceOrder")
    channels = (f"bbo.{config.paradex_market}",
                f"order_book.{config.paradex_market}.snapshot@15@50ms")
    sources = [
        SourceSpec("binance_spot", "wss://stream.binance.com:9443/stream?streams=" + "/".join(spot),
                   config.symbol, spot, depth_url="https://api.binance.com/api/v3/depth?" +
                   urlencode({"symbol": config.symbol, "limit": 1000})),
        SourceSpec("binance_perp_book", "wss://fstream.binance.com/public/stream?streams=" + "/".join(perp_book),
                   config.symbol, perp_book, depth_url="https://fapi.binance.com/fapi/v1/depth?" +
                   urlencode({"symbol": config.symbol, "limit": 1000}), futures=True),
        SourceSpec("binance_perp_market", "wss://fstream.binance.com/market/stream?streams=" + "/".join(perp_market),
                   config.symbol, perp_market, futures=True),
        SourceSpec("paradex", "wss://ws.api.prod.paradex.trade/v1?sbeSchemaId=1&sbeSchemaVersion=1",
                   config.paradex_market, channels,
                   tuple({"jsonrpc": "2.0", "method": "subscribe", "params": {"channel": channel}, "id": index}
                         for index, channel in enumerate(channels, 1))),
    ]
    return [source for source in sources if source.name in config.sources]
