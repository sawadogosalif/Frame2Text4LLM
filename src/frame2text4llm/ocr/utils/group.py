import re
import pandas as pd
from datetime import timedelta
from collections import Counter

def group_and_clean_text(data, signature_keywords=None, window_sec=30):
    """
    Regroupe les textes en tranches de temps et nettoie les signatures de façon
    à supprimer tous les mots contenant la base de signature (ex. "clideo").
    
    Args:
        data (list[dict]): Liste de dicts avec 'time_formatted' (MM:SS:ms) et 'text'.
        signature_keywords (str | list[str] | None):
            - None => détection auto (>95% des textes).
            - str  => base de signature (ex. "clideo").
            - list => listes de signatures ou bases (ex. ["clideo", "autre"]).
        window_sec (int): Durée de la fenêtre de grouping en secondes.
    
    Returns:
        list[dict]: Chaque élément contient 'start_time', 'end_time', 'aggregated_text'.
    """
    df = pd.DataFrame(data)
    df['text'] = df['text'].astype(str)
    
    # 1) Détection auto si nécessaire
    if signature_keywords is None:
        word_lists = df['text'].str.lower().str.split()
        counts = Counter(w for words in word_lists for w in set(words))
        n = len(df)
        # on garde les mots présents >95% des lignes
        signature_keywords = [w for w,c in counts.items() if c/n > 0.9]
    
    # 2) Normalisation du paramètre en liste de str
    if isinstance(signature_keywords, str):
        signature_keywords = [signature_keywords]
    elif not isinstance(signature_keywords, list):
        signature_keywords = []
    
    # 3) Construction du pattern regex
    patterns = []
    for kw in signature_keywords:
        # si la kw contient un point, on l'échappe et on supprime exactement ce token
        if '.' in kw:
            esc = re.escape(kw)
            patterns.append(rf'\b{esc}\b')
        else:
            # base sans point => supprime kw + .suffix ou kwXYZ
            esc = re.escape(kw)
            patterns.append(rf'\b{esc}(?:\.\S+)?\b')
    combined_pattern = '|'.join(patterns) if patterns else r'a^'  # 'a^' pour aucun match
    
    df['clean_text'] = df['text'].str.replace(combined_pattern, '', flags=re.IGNORECASE, regex=True)
    df['clean_text'] = df['clean_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    parts = df['time_formatted'].str.extract(r'(?P<mm>\d+):(?P<ss>\d+):(?P<ms>\d+)')
    df['time'] = (
        pd.to_timedelta(parts['mm'].astype(int), unit='m') +
        pd.to_timedelta(parts['ss'].astype(int), unit='s') +
        pd.to_timedelta(parts['ms'].astype(int), unit='ms')
    )
    
    t0 = df['time'].min()
    df['group'] = ((df['time'] - t0) / timedelta(seconds=window_sec)).astype(int)
    
    agg = df.groupby('group').agg({
        'time': ['min', 'max'],
        'clean_text': lambda texts: ' '.join(texts).strip()
    }).reset_index(drop=True)
    agg.columns = ['start_td', 'end_td', 'aggregated_text']
    
    agg['start_time'] = agg['start_td'].apply(lambda td: str(td).split('.')[0])
    agg['end_time']   = agg['end_td'].apply(lambda td: str(td).split('.')[0])
    
    return agg[['start_time', 'end_time', 'aggregated_text']].to_dict(orient='records')