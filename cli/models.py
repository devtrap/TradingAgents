from enum import Enum


class AnalystType(str, Enum):
    MARKET = "market"
    # Wire value stays "social" for saved-config and string-keyed-caller
    # back-compat; the user-facing label is "Sentiment Analyst".
    SOCIAL = "social"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"


class AssetType(str, Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"


# Instruments with no issuing company, so no balance sheet, cash flow, income
# statement or insider filings exist. Running the Fundamentals Analyst on these
# burns ~4 LLM calls to reach a NO_DATA sentinel, then feeds "fundamentals
# unavailable" into every downstream debate prompt.
NON_EQUITY_ASSET_TYPES = frozenset({
    AssetType.CRYPTO,
    AssetType.FOREX,
    AssetType.COMMODITY,
    AssetType.INDEX,
})
