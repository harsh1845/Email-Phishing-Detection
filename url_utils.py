import re
from urllib.parse import urlparse
import tldextract
import pandas as pd

def count_sensitive_keywords(url):
    """Counts sensitive keywords in a URL."""
    keywords = ['login', 'signin', 'secure', 'account', 'update', 'verify', 'bank', 'paypal', 'confirm']
    count = 0
    if isinstance(url, str):
        for keyword in keywords:
            if keyword in url.lower():
                count += 1
    return count

def extract_url_features(url):
    """
    Extracts a dictionary of features from a single URL.
    This version is more robust against malformed URLs.
    """
    default_features = {
        'has_ip_address': 0, 'raw_url_length': 0, 'raw_hostname_length': 0,
        'has_at_symbol': 0, 'has_double_slash_redirect': 0, 'has_hyphen_in_domain': 0,
        'subdomain_cat': 0, 'sensitive_keyword_count': 0
    }
    
    if not isinstance(url, str) or not url:
        return pd.Series(default_features)

    try:
        features = {}
        
        # We parse the URL once at the beginning
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        # Feature extraction logic
        ip_parts = domain.split('.')
        features['has_ip_address'] = 1 if len(ip_parts) == 4 and all(part.isdigit() for part in ip_parts) else 0

        features['raw_url_length'] = len(url)
        features['raw_hostname_length'] = len(domain)
        features['has_at_symbol'] = 1 if '@' in url else 0
        features['has_double_slash_redirect'] = 1 if url.rfind('//') > 7 else 0
        features['has_hyphen_in_domain'] = 1 if '-' in domain else 0

        ext = tldextract.extract(url)
        subdomains = [s for s in ext.subdomain.split('.') if s and s.lower() != 'www']
        num_dots = len(subdomains)
        if num_dots == 0:
            features['subdomain_cat'] = 0
        elif num_dots == 1:
            features['subdomain_cat'] = 1
        else:
            features['subdomain_cat'] = 2
            
        features['sensitive_keyword_count'] = count_sensitive_keywords(url)
        
        return pd.Series(features)

    except ValueError:
        # If urlparse or tldextract fails (e.g., "Invalid IPv6 URL"), return defaults
        return pd.Series(default_features)