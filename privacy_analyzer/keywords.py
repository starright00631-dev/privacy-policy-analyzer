# keywords.py
# Keyword dictionaries for privacy policy classification.
# Each category uses phrases found in real privacy policies.

DATA_CATEGORIES = {
    "Location": [
        "location", "gps", "geolocation", "latitude", "longitude",
        "ip address", "geographic", "whereabouts", "region", "precise location"
    ],
    "Financial": [
        "payment", "credit card", "bank account", "billing", "transaction",
        "purchase history", "financial information", "payment method"
    ],
    "Biometric": [
        "biometric", "fingerprint", "face recognition", "facial recognition",
        "iris scan", "retina", "voice recognition", "physiological"
    ],
    "Browsing & Device": [
        "cookies", "browsing history", "search history", "device identifier",
        "browser type", "operating system", "clickstream", "log data",
        "page views", "referral url", "device information"
    ],
    "Contact & Identity": [
        "full name", "email address", "phone number", "mailing address",
        "date of birth", "social security", "passport", "government-issued id",
        "username", "account name"
    ],
    "Behavioral & Inferred": [
        "preferences", "interests", "inferences", "behavioral data",
        "usage patterns", "interactions", "profile", "segments", "characteristics"
    ],
    "Communications": [
        "messages", "emails", "calls", "chat history", "correspondence",
        "communications content", "customer support"
    ]
}

# Language that gives users control (positive signals)
OPT_OUT_PHRASES = [
    "opt out", "opt-out", "do not sell", "do not share",
    "withdraw consent", "right to object", "right to erasure",
    "right to deletion", "right to be forgotten", "unsubscribe",
    "you may request", "you can request", "contact us to"
]

# Language where consent is assumed or buried (negative signals)
OPT_IN_PHRASES = [
    "by using", "by accessing", "by continuing", "you agree",
    "you consent", "deemed to have accepted", "acceptance of",
    "your use constitutes", "continued use"
]

# Third-party sharing indicators
THIRD_PARTY_PHRASES = [
    "third party", "third-party", "our partners", "our affiliates",
    "advertising partners", "service providers", "vendors",
    "share with", "sell your data", "sell your information",
    "disclose to", "transfer to"
]

# Data retention language
RETENTION_PHRASES = [
    "retain", "retention period", "store for", "keep for",
    "delete after", "deletion policy", "as long as necessary",
    "indefinitely", "until you close", "until you request"
]
