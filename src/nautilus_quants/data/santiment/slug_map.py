# Copyright (c) 2025 nautilus_quants
# SPDX-License-Identifier: MIT
"""
Santiment slug mapping — verified 2026-04-17 against SANBASE PRO.

Maps crypto tickers to Santiment project slugs for san.get() queries.
Slugs resolved via allProjects GraphQL query.

Multi-candidate ticker selection rule: SanAPI stores funding_rate /
open_interest per-slug (NOT aggregated across a ticker's multiple
projects). For tickers with several slugs sharing a ticker symbol
(e.g. STRK, TON, TRUMP, WLD, ZRO, FLOKI, GTC, ORDI, LINA), we pick
the slug that actually has non-empty funding_rate data — which is
sometimes the chain-specific variant (WLD → o-worldcoin-org) and
sometimes a legacy sibling project that SanAPI reuses as the data
anchor (STRK → strike, TRUMP → maga).
"""
from __future__ import annotations

# ── Ticker → Santiment slug (all verified mappings) ───────────────────
SLUG_MAP: dict[str, str] = {
    "1INCH": "1inch",
    "AAVE": "aave",
    "ADA": "cardano",
    "AGLD": "adventure-gold",
    "AIXBT": "aixbt",
    "ALGO": "algorand",
    "ALICE": "myneighboralice",
    "ALPHA": "alpha-finance-lab",
    "ANIME": "anime",
    "ANKR": "ankr",
    "ANT": "aragon",
    "APT": "aptos",
    "AR": "arweave",
    "ARB": "arb-arbitrum",
    "ARKM": "arkham",
    "ARPA": "arpa-chain",
    "ATA": "automata-network",
    "ATOM": "cosmos",
    "AUDIO": "audius",
    "AVAX": "avalanche",
    "AXS": "axie-infinity",
    "BABY": "babylon",
    "BAKE": "bakerytoken",
    "BAL": "balancer",
    "BAND": "band-protocol",
    "BAT": "basic-attention-token",
    "BCH": "bitcoin-cash",
    "BEL": "bella-protocol",
    "BERA": "berachain",
    "BLZ": "bluzelle",
    "BNB": "binance-coin",
    "BOME": "book-of-meme",
    "BONK": "bonk1",
    "BTC": "bitcoin",
    "C98": "coin98",
    "CELO": "celo",
    "CELR": "celer-network",
    "CHR": "chromia",
    "CHZ": "chiliz",
    "COMP": "compound",
    "COTI": "coti",
    "CRV": "arb-curve",                   # SanAPI stores CRV FR/OI here (vs curve which has no FR/OI)
    "CTSI": "cartesi",
    "DASH": "dash",
    "DENT": "dent",
    "DGB": "digibyte",
    "DOGE": "dogecoin",
    "DOT": "polkadot-new",
    "DUSK": "dusk-network",
    "DYDX": "dydx",
    "EGLD": "elrond-egld",
    "EIGEN": "eigenlayer",
    "ENJ": "enjin-coin",
    "ENS": "ethereum-name-service",
    "ETC": "ethereum-classic",
    "ETH": "ethereum",
    "ETHFI": "ether-fi-ethfi",
    "FIL": "file-coin",
    "FLM": "flamingo",
    "FLOKI": "bnb-floki-inu",             # SanAPI stores FLOKI FR/OI here (vs floki-inu-v2 which has no FR/OI)
    "FTM": "fantom",
    "GALA": "gala-v2",
    "GRT": "the-graph",
    "GTC": "game",                        # SanAPI stores GTC (Gitcoin) FR/OI under `game` slug (vs gitcoin which has no FR/OI)
    "HBAR": "hedera-hashgraph",
    "HOT": "holo",
    "HUMA": "sol-huma-finance",
    "ICX": "icon",
    "IOST": "iostoken",
    "IOTA": "iota",
    "IOTX": "iotex",
    "JTO": "jito",
    "JUP": "jupiter-ag",
    "KAVA": "kava",
    "KLAY": "klaytn",
    "KSM": "kusama",
    "LINA": "linear",
    "LINK": "chainlink",
    "LPT": "livepeer",
    "LRC": "loopring",
    "LTC": "litecoin",
    "MANA": "decentraland",
    "MASK": "mask-network",
    "MATIC": "polygon-ecosystem-token",
    "MEME": "meme",
    "MKR": "maker",
    "MORPHO": "morpho",
    "MTL": "metal",
    "NEAR": "near-protocol",
    "NEO": "neo",
    "NKN": "nkn",
    "OCEAN": "ocean-protocol",
    "OGN": "origin-protocol",
    "OMG": "omisego",
    "ONDO": "ondo-finance",
    "ONE": "harmony",
    "ONT": "ontology",
    "ORDI": "ordi",                       # ORDI token (vs ordinals inscriptions)
    "PENGU": "pudgy-penguins",
    "PEOPLE": "constitutiondao",
    "PEPE": "pepe",
    "PNUT": "peanut-the-squirrel",
    "PYTH": "pyth-network",
    "QTUM": "qtum",
    "REEF": "reef",
    "REN": "ren",
    "RENDER": "render",
    "RESOLV": "resolv",
    "RLC": "rlc",
    "ROSE": "oasis-network",
    "RSR": "reserve-rights",
    "RUNE": "thorchain",
    "RVN": "ravencoin",
    "SAHARA": "sahara-ai",
    "SAND": "the-sandbox",
    "SATS": "sats-ordinals",
    "SFP": "bnb-safepal",
    "SHIB": "shiba-inu",
    "SKL": "skale-network",
    "SNX": "synthetix-network-token",
    "SOL": "solana",
    "STMX": "stormx",
    "STORJ": "storj",
    "STRK": "strike",                     # SanAPI stores STRK (Starknet) FR/OI under `strike` slug (vs starknet-token which has no FR/OI)
    "SUI": "sui",
    "SUSHI": "arb-sushi",                 # SanAPI stores SUSHI FR/OI here (vs sushi which has no FR/OI)
    "SXP": "swipe",
    "THETA": "theta",
    "TIA": "celestia",
    "TON": "toncoin",                     # Toncoin (vs tontoken)
    "TRB": "tellor",
    "TRUMP": "maga",                      # SanAPI stores TRUMP FR/OI under `maga` slug (vs official-trump which has no FR/OI)
    "TRX": "tron",
    "UNI": "p-uniswap",                   # Polygon variant — SanAPI stores UNI FR/OI here (vs uniswap which has no FR/OI)
    "VET": "vechain",
    "VIRTUAL": "virtual-protocol",
    "W": "wormhole",
    "WAVES": "waves",
    "WIF": "dogwifhat",
    "WLD": "o-worldcoin-org",             # Optimism variant — SanAPI stores WLD FR/OI here (vs worldcoin-org which has no FR/OI)
    "WLFI": "world-liberty-financial-wlfi",
    "XEC": "ecash",
    "XEM": "nem",
    "XLM": "stellar",
    "XMR": "monero",
    "XRP": "xrp",
    "XTZ": "tezos",
    "YFI": "yearn-finance",
    "ZEC": "zcash",
    "ZEN": "zencash",
    "ZIL": "zilliqa",
    "ZK": "zksync",
    "ZRO": "arb-layerzero",               # Arbitrum variant — SanAPI stores ZRO FR/OI here (vs layerzero which has no FR/OI)
    "ZRX": "0x",
}

# ── Tickers confirmed NOT available in Santiment (2026-04-17) ─────────
# Listed here so data_santiment.yaml maintainers know to exclude them.
UNAVAILABLE_IN_SANTIMENT: frozenset[str] = frozenset({
    "DEFI",   # Binance futures "DEFI index" — not a real Santiment project
})

# ── Tickers with a valid slug but no / partial FR+OI in SanAPI ─────────
# Verified 2026-04-17 across every allProjects candidate slug, 4h bars
# over 2025-09-01 → 2026-03-15 (1176 bars expected).
# These still have volume_usd / social_volume_total, so they are kept in
# SLUG_MAP. Strategies depending on san_funding_rate / san_open_interest
# should expect NaN or stale values for these tickers.
FR_OI_MISSING_IN_SANTIMENT: frozenset[str] = frozenset({
    # ── FR + OI both empty (0 rows) ──
    # Older Binance tickers whose derivatives data SanAPI has dropped:
    "ADA", "ALICE", "ANT", "ATA", "GALA", "KLAY", "LINA", "OCEAN", "OMG",
    "REEF", "SNX", "STMX", "XEC",
    # 2024-2025 new coins SanAPI has not backfilled derivatives for:
    "HUMA", "BABY", "MEME", "BONK", "SAHARA",
    # FR available but OI dropped:
    "BLZ", "FTM",
    # ── Partial FR coverage (< ~85% of window) ──
    # Last-30-bars only (≈5 days at 4h) — new coins SanAPI just started:
    "ETHFI", "PNUT", "ANIME", "AGLD", "ARKM", "W",
    # Older tickers SanAPI partially rebackfilled:
    "C98", "PEOPLE",
    # Older tickers under active drop (69 / 255 / 598 rows):
    "BAL", "REN", "XEM", "ALPHA",
})

# ── Per-metric availability (verified 2026-04-12, 4h, 1yr) ───────────

FR_AVAILABLE: frozenset[str] = frozenset(
    {
        "1INCH",
        "AAVE",
        "ALGO",
        "ALPHA",
        "ANKR",
        "AR",
        "ARPA",
        "ATOM",
        "AUDIO",
        "AVAX",
        "AXS",
        "BAKE",
        "BAL",
        "BAND",
        "BAT",
        "BCH",
        "BEL",
        "BLZ",
        "BNB",
        "BTC",
        "C98",
        "CELO",
        "CELR",
        "CHR",
        "CHZ",
        "COMP",
        "COTI",
        "CTSI",
        "DASH",
        "DENT",
        "DGB",
        "DOGE",
        "DOT",
        "DUSK",
        "DYDX",
        "ENJ",
        "ENS",
        "ETC",
        "ETH",
        "FTM",
        "GRT",
        "HBAR",
        "HOT",
        "ICX",
        "IOST",
        "IOTA",
        "IOTX",
        "KAVA",
        "KSM",
        "LINK",
        "LPT",
        "LRC",
        "LTC",
        "MANA",
        "MASK",
        "MATIC",
        "MKR",
        "MTL",
        "NEAR",
        "NEO",
        "NKN",
        "OGN",
        "OMG",
        "ONE",
        "ONT",
        "PEOPLE",
        "QTUM",
        "ROSE",
        "RSR",
        "RUNE",
        "RVN",
        "SAND",
        "SHIB",
        "SOL",
        "STORJ",
        "SXP",
        "TRB",
        "TRX",
        "VET",
        "WAVES",
        "XEM",
        "XLM",
        "XMR",
        "XRP",
        "XTZ",
        "YFI",
        "ZEC",
        "ZIL",
        "ZRX",
    }
)  # 89 tickers

OI_AVAILABLE: frozenset[str] = frozenset(
    {
        "1INCH",
        "AAVE",
        "ADA",
        "ALGO",
        "ALPHA",
        "ANKR",
        "AR",
        "ARPA",
        "ATOM",
        "AUDIO",
        "AVAX",
        "AXS",
        "BAKE",
        "BAL",
        "BAND",
        "BAT",
        "BCH",
        "BEL",
        "BNB",
        "BTC",
        "C98",
        "CELO",
        "CELR",
        "CHR",
        "CHZ",
        "COMP",
        "COTI",
        "CRV",
        "CTSI",
        "DASH",
        "DENT",
        "DGB",
        "DOGE",
        "DOT",
        "DUSK",
        "DYDX",
        "ENJ",
        "ENS",
        "ETC",
        "ETH",
        "GRT",
        "HBAR",
        "HOT",
        "ICX",
        "IOST",
        "IOTA",
        "IOTX",
        "KAVA",
        "KSM",
        "LINK",
        "LPT",
        "LRC",
        "LTC",
        "MANA",
        "MASK",
        "MATIC",
        "MKR",
        "MTL",
        "NEAR",
        "NEO",
        "NKN",
        "OGN",
        "OMG",
        "ONE",
        "ONT",
        "PEOPLE",
        "QTUM",
        "ROSE",
        "RSR",
        "RUNE",
        "RVN",
        "SAND",
        "SHIB",
        "SNX",
        "SOL",
        "STORJ",
        "SUSHI",
        "SXP",
        "TRB",
        "TRX",
        "UNI",
        "VET",
        "WAVES",
        "XEM",
        "XLM",
        "XMR",
        "XRP",
        "XTZ",
        "YFI",
        "ZEC",
        "ZIL",
        "ZRX",
    }
)  # 92 tickers

# FR ∩ OI — default download/mining universe
AVAILABLE: frozenset[str] = FR_AVAILABLE & OI_AVAILABLE  # 87 tickers


def ticker_to_slug(ticker: str) -> str | None:
    """Map a plain ticker (e.g. ``"BTC"``) to a Santiment slug.

    Returns ``None`` if the ticker has no verified mapping.
    """
    return SLUG_MAP.get(ticker.upper())


def instrument_to_slug(instrument_id: str) -> str | None:
    """Map an instrument ID (e.g. ``"BTCUSDT.BINANCE"``) to a Santiment slug.

    Strips the ``USDT`` suffix and venue, handles ``1000`` prefix (e.g.
    ``"1000SHIBUSDT.BINANCE"`` → ``"SHIB"`` → ``"shiba-inu"``).
    """
    symbol = instrument_id.split(".")[0]
    symbol = symbol.replace("USDT", "")
    if symbol.startswith("1000"):
        symbol = symbol[4:]
    return ticker_to_slug(symbol)


def instrument_to_ticker(instrument_id: str) -> str:
    """Extract plain ticker from instrument ID.

    ``"BTCUSDT.BINANCE"`` → ``"BTC"``,
    ``"1000SHIBUSDT.BINANCE"`` → ``"SHIB"``.
    """
    symbol = instrument_id.split(".")[0].replace("USDT", "")
    if symbol.startswith("1000"):
        symbol = symbol[4:]
    return symbol.upper()
