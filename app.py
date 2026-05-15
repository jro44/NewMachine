import streamlit as st
import pandas as pd
import PyPDF2
import re
import numpy as np
from collections import Counter

# ==========================================
# KONFIGURACJA GŁÓWNA
# ==========================================
st.set_page_config(page_title="Silnik Analityczny Eurojackpot", page_icon="⚙️", layout="wide")

# Ustawienie na sztywno: analizujemy 50 ostatnich losowań
# Zbyt długa historia wprowadza szum mechaniczny starych bębnów maszyny.
N_DRAWS_TO_ANALYZE = 50 

# ==========================================
# MODUŁ EKSTRAKCJI DANYCH (PARSER BUFOROWY)
# ==========================================
@st.cache_data
def extract_draw_data(pdf_path, expected_balls, max_ball_val):
    """
    Kuloodporny parser tokenów. Ignoruje wizualną siatkę PDF.
    Opiera się na logice: identyfikatory losowań to 4-cyfrowe liczby (np. '0954').
    Zbiera kule do bufora i rozdziela je pomiędzy oczekujące losowania.
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

        # Ekstrakcja wszystkich ciągów cyfr
        tokens = re.findall(r'\b\d+\b', text)
        
        pending_draws = []
        collected_balls = []
        data = []
        
        for t in tokens:
            val = int(t)
            
            # Detekcja ID losowania: 4 cyfry, omijamy lata ze stopek (np. 2024, 2026)
            is_draw_id = len(t) == 4 and (t.startswith('0') or val < 1500)
            
            if is_draw_id:
                pending_draws.append(val)
            elif 1 <= val <= max_ball_val:
                # Jeśli mamy oczekujące losowania, zbieramy kule
                if pending_draws:
                    collected_balls.append(val)
                    
                    # Sprawdzamy, czy bufor kul wypełnił zapotrzebowanie wszystkich oczekujących losowań
                    if len(collected_balls) == len(pending_draws) * expected_balls:
                        # Rozdzielamy kule do losowań
                        for i, draw_id in enumerate(pending_draws):
                            # Wycinamy odpowiednią partię kul z bufora
                            start_idx = i * expected_balls
                            end_idx = start_idx + expected_balls
                            draw_balls = collected_balls[start_idx:end_idx]
                            
                            # Sortujemy i dodajemy do danych
                            unique_balls = sorted(list(set(draw_balls)))
                            if len(unique_balls) == expected_balls:
                                data.append([draw_id] + unique_balls)
                        
                        # Czyścimy bufory po udanym przypisaniu
                        pending_draws = []
                        collected_balls = []
                        
        if not data:
            return pd.DataFrame()
            
        # Konwersja na DataFrame
        cols = ['Losowanie'] + [f'Kula_{i+1}' for i in range(expected_balls)]
        df = pd.DataFrame(data, columns=cols)
        
        # Sortowanie od najnowszego i usunięcie duplikatów
        df = df.sort_values(by='Losowanie', ascending=False).drop_duplicates(subset=['Losowanie']).reset_index(drop=True)
        return df
        
    except FileNotFoundError:
        st.error(f"Nie znaleziono pliku: {pdf_path}. Upewnij się, że jest w tym samym folderze co skrypt.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Błąd krytyczny parsera dla pliku {pdf_path}: {e}")
        return pd.DataFrame()

# ==========================================
# SILNIK ANALITYCZNY I OBLICZENIOWY
# ==========================================
def analyze_statistics(df):
    """
    Oblicza różnice (skoki) pomiędzy kolejnymi losowaniami oraz wyciąga gorące liczby.
    Używamy wektora różnic (signed difference), aby śledzić kierunek ruchu maszyny.
    """
    recent_df = df.head(N_DRAWS_TO_ANALYZE).copy()
    
    # Różnica z poprzednim chronologicznie losem: nowszy (wiersz i) - starszy (wiersz i+1)
    # Wartość dodatnia oznacza, że nowa kula była wyższa.
    movements = recent_df.set_index('Losowanie').diff(periods=-1).dropna().astype(int)
    
    # Obliczanie gorących liczb
    all_balls = recent_df.iloc[:, 1:].values.flatten()
    hot_digits = Counter(all_balls).most_common()
    
    return recent_df, movements, hot_digits

def silver_bullet_generator(recent_df, movements, hot_digits_list, pool_size, expected_balls):
    """
    Silver Bullet: Pobiera najczęstszy kierunkowy ruch maszyny (modę) dla każdej kuli,
    aplikuje go na ostatnie losowanie. W razie kolizji wspiera się gorącymi liczbami.
    """
    if recent_df.empty or movements.empty:
        return []

    # Ostatnie fizyczne losowanie
    last_draw = recent_df.iloc[0, 1:].values
    
    # Wyliczanie najczęstszego ruchu (mody) dla poszczególnych komór
    likely_movements = []
    for col in movements.columns:
        # Pobieramy modę (jeśli jest ich kilka, bierzemy pierwszą)
        mode_val = int(movements[col].mode().iloc[0])
        likely_movements.append(mode_val)
    
    generated_set = set()
    
    # Krok 1: Aplikacja wyliczonej trajektorii maszyny na ostatnie losowanie
    for ball, delta in zip(last_draw, likely_movements):
        proposed_ball = ball + delta
        
        # Walidacja: czy kula mieści się w bębnie i czy nie jest duplikatem
        if 1 <= proposed_ball <= pool_size and proposed_ball not in generated_set:
            generated_set.add(proposed_ball)
        else:
            # Mechanizm ratunkowy: jeśli ruch wykracza poza bęben, odbijamy wektor w drugą stronę
            alternative_ball = ball - delta
            if 1 <= alternative_ball <= pool_size and alternative_ball not in generated_set:
                generated_set.add(alternative_ball)
                
    # Krok 2: Wypełnianie braków (jeśli wektory spowodowały nałożenie się kul na siebie)
    if len(generated_set) < expected_balls:
        for num, _ in hot_digits_list:
            if num not in generated_set:
                generated_set.add(num)
            if len(generated_set) == expected_balls:
                break
                
    return sorted(list(generated_set))

# ==========================================
# INTERFEJS UŻYTKOWNIKA (UI)
# ==========================================
st.title("🎯 Eurojackpot: Zaawansowany Silnik Analityczny")
st.markdown("Zaprojektowane by **A.K.** | Moduł Analizy Wektorowej i Mechaniki Losowań")

# Wczytywanie i Parsowanie Danych
with st.spinner("Inicjalizacja parsera tokenów... Wczytywanie map Multipasko..."):
    df_5z50 = extract_draw_data("5z50.PDF", 5, 50)
    df_2z12 = extract_draw_data("2z12.PDF", 2, 12)

if df_5z50.empty or df_2z12.empty:
    st.error("Brak poprawnych danych. Upewnij się, że pliki 5z50.PDF i 2z12.PDF są w folderze aplikacji.")
else:
    # Sukces parsowania - pokazujemy ostatnie zapisane losowanie by udowodnić, że parser działa bezbłędnie
    last_draw_id = df_5z50['Losowanie'].iloc[0]
    st.success(f"Analiza zakończona sukcesem. Zdekodowano historię. Ostatnie zmapowane losowanie: **{last_draw_id}**")
    
    # Procesowanie Statystyk
    recent_5z50, movements_5z50, hot_5z50 = analyze_statistics(df_5z50)
    recent_2z12, movements_2z12, hot_2z12 = analyze_statistics(df_2z12)

    # --- Wizualizacja Danych ---
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bęben 1: Statystyki 5/50")
        st.markdown(f"*Analiza odchyleń z ostatnich {N_DRAWS_TO_ANALYZE} losowań*")
        st.dataframe(movements_5z50.style.background_gradient(cmap='RdYlGn', axis=0), height=300, use_container_width=True)
        st.markdown("**Top 10 Gorących Liczb:**")
        st.code(" | ".join([f"{n} ({c}x)" for n, c in hot_5z50[:10]]))

    with col2:
        st.subheader("Bęben 2: Statystyki 2/12")
        st.markdown(f"*Analiza odchyleń z ostatnich {N_DRAWS_TO_ANALYZE} losowań*")
        st.dataframe(movements_2z12.style.background_gradient(cmap='RdYlGn', axis=0), height=300, use_container_width=True)
        st.markdown("**Top 5 Gorących Liczb:**")
        st.code(" | ".join([f"{n} ({c}x)" for n, c in hot_2z12[:5]]))

    st.divider()

    # --- Generacja Srebrnej Kuli ---
    st.header("⚡ Funkcja 'Silver Bullet'")
    st.markdown("""
    To nie jest ślepy generator `random`. Algorytm wylicza **modę skoku** (najczęściej występującą matematyczną różnicę pomiędzy poprzednim a aktualnym losem) oddzielnie dla każdej z 7 wylosowanych pozycji. 
    Następnie nakłada te wektory ruchu na absolutnie ostatnie losowanie. Ewentualne braki uzupełnia najgorętszymi kulami z puli.
    """)
    
    # Renderowanie wyjścia generatora
    if st.button("🚀 Uruchom Analizę i Generuj Zestaw", type="primary", use_container_width=True):
        silver_5 = silver_bullet_generator(recent_5z50, movements_5z50, hot_5z50, 50, 5)
        silver_2 = silver_bullet_generator(recent_2z12, movements_2z12, hot_2z12, 12, 2)
        
        # Formatowanie do eleganckiego widoku
        str_5 = " 🟢 ".join([f"{x:02d}" for x in silver_5])
        str_2 = " 🟡 ".join([f"{x:02d}" for x in silver_2])
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("### Wytypowany zestaw na podstawie mechaniki i statystyki:")
        st.success(f"## {str_5}  ➕  {str_2}")

       
if __name__ == "__main__":
    main()
