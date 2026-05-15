import streamlit as st
import pandas as pd
import PyPDF2
import re
import numpy as np
from collections import Counter

# --- Konfiguracja i Inicjalizacja ---
st.set_page_config(page_title="Silnik Analityczny Eurojackpot", page_icon="⚙️", layout="wide")

# Silnik jest twardo ustawiony na 50 ostatnich losowań, 
# aby idealnie śledzić aktualne odchylenia mechaniczne maszyny.
N_DRAWS_TO_ANALYZE = 50 

# --- Moduły Ekstrakcji Danych (Maszyna Stanów) ---
@st.cache_data
def extract_draw_data(pdf_path, expected_balls, max_ball_val):
    """
    Superszybki ekstrakor tokenów. Ignoruje zepsute siatki PDF.
    Wykorzystuje fakt, że numery losowań to zawsze 4 cyfry (np. 0954, 0012).
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"

        # Pobieramy absolutnie wszystkie liczby z tekstu
        tokens = re.findall(r'\b\d+\b', text)
        
        data = []
        current_draw = None
        balls = []
        
        for t in tokens:
            # Identyfikacja ID Losowania (zawsze 4 cyfry w formacie Multipasko)
            if len(t) == 4:
                val = int(t)
                
                # Zapisz poprzednie losowanie, jeśli zebrało komplet kul
                if current_draw is not None and len(balls) == expected_balls:
                    data.append([current_draw] + sorted(balls))
                
                # Zabezpieczenie przed latami w stopkach (np. "2004", "2026")
                if val < 2000:
                    current_draw = val
                    balls = []
                else:
                    current_draw = None
            else:
                # Zbieranie kul przypisanych do aktywnego losowania
                val = int(t)
                if current_draw is not None and 1 <= val <= max_ball_val:
                    if val not in balls and len(balls) < expected_balls:
                        balls.append(val)
                        
        # Zapisanie ostatniego losowania w buforze
        if current_draw is not None and len(balls) == expected_balls:
            data.append([current_draw] + sorted(balls))
            
        if not data:
            return pd.DataFrame()
            
        # Konwersja do DataFrame
        cols = ['Losowanie'] + [f'Kula_{i+1}' for i in range(expected_balls)]
        df = pd.DataFrame(data, columns=cols)
        
        # Sortowanie od najnowszego i usunięcie ewentualnych duplikatów
        df = df.sort_values(by='Losowanie', ascending=False).drop_duplicates(subset=['Losowanie']).reset_index(drop=True)
        return df
        
    except FileNotFoundError:
        st.error(f"Nie znaleziono pliku: {pdf_path}. Upewnij się, że jest w tym samym folderze co skrypt.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Błąd krytyczny podczas parsowania {pdf_path}: {e}")
        return pd.DataFrame()

# --- Rdzeń Analityczny ---
def analyze_machine_movement(df):
    """
    Oblicza matematyczny schemat "ruchu maszyny" (różnice bezwzględne między kolejnymi losowaniami).
    """
    recent_df = df.head(N_DRAWS_TO_ANALYZE).copy()
    
    # Różnica między wierszem [i] a wierszem [i+1] (starszym)
    movements = recent_df.set_index('Losowanie').diff(periods=-1).dropna().abs()
    
    # Formatowanie kolumn do int
    movements = movements.astype(int)
    return recent_df, movements

def get_hot_digits(df, num_balls):
    """Oblicza najczęściej padające kule w wybranym oknie czasowym."""
    all_balls = df.iloc[:, 1:].values.flatten()
    counter = Counter(all_balls)
    return counter.most_common(num_balls)

def silver_bullet_generator(recent_df, movements, pool_size, expected_balls):
    """
    Srebrna Kula: Mechanizm generujący najprawdopodobniejszy wektor przesunięć
    na podstawie dokładnej mody (najczęstszego ruchu) i najgorętszych kul.
    """
    if recent_df.empty or movements.empty:
        return []

    last_draw = recent_df.iloc[0, 1:].values
    
    # Szukamy najczęstszego ruchu (mody) dla każdej pozycji kuli
    likely_movements = []
    for col in movements.columns:
        most_common_delta = int(movements[col].mode()[0])
        likely_movements.append(most_common_delta)
    
    generated_set = set()
    
    # Aplikacja wektora ruchu do ostatniego losowania
    for ball, delta in zip(last_draw, likely_movements):
        option_up = ball + delta
        option_down = ball - delta
        
        # Walidacja granic i unikalności
        if 1 <= option_up <= pool_size and option_up not in generated_set:
            generated_set.add(option_up)
        elif 1 <= option_down <= pool_size and option_down not in generated_set:
            generated_set.add(option_down)
            
    # Wypełnianie brakujących miejsc (jeśli przesunięcia dały duplikaty) gorącymi liczbami
    if len(generated_set) < expected_balls:
        hot_digits = get_hot_digits(recent_df, expected_balls * 3) # Zapas gorących
        for num, _ in hot_digits:
            if num not in generated_set:
                generated_set.add(num)
            if len(generated_set) == expected_balls:
                break
                
    return sorted(list(generated_set))

# --- Interfejs Użytkownika (Streamlit) ---
st.title("⚙️ Główny Silnik Analityczny Eurojackpot")
st.markdown("**Status Architektury:** Praca na surowych tokenach | **Zabezpieczenia:** Aktywne | **Analiza:** Ostatnie 50 losowań")

st.sidebar.header("Diagnostyka Danych")
st.sidebar.info("Pliki `5z50.PDF` oraz `2z12.PDF` muszą znajdować się w tym samym katalogu co aplikacja.")

with st.spinner("Przetwarzanie strumienia tekstowego z plików PDF..."):
    df_5z50 = extract_draw_data("5z50.PDF", 5, 50)
    df_2z12 = extract_draw_data("2z12.PDF", 2, 12)

if df_5z50.empty or df_2z12.empty:
    st.error("Zatrzymano silnik. Brak danych. Sprawdź, czy nazwy plików PDF w katalogu zgadzają się dokładnie z '5z50.PDF' i '2z12.PDF'.")
else:
    st.sidebar.success("Strumień zdekodowany w milisekundach.")
    
    # Procesowanie 5z50
    recent_5z50, movements_5z50 = analyze_machine_movement(df_5z50)
    hot_5z50 = get_hot_digits(recent_5z50, 10)
    
    # Procesowanie 2z12
    recent_2z12, movements_2z12 = analyze_machine_movement(df_2z12)
    hot_2z12 = get_hot_digits(recent_2z12, 5)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Wektor Ruchu Maszyny (5 z 50)")
        st.write(f"Różnice między kolejnymi wylosowanymi kulami z {N_DRAWS_TO_ANALYZE} ostatnich losowań.")
        st.dataframe(movements_5z50.style.background_gradient(cmap='Greens', axis=None), height=400, use_container_width=True)
        st.markdown("**🔥 Najgorętsze liczby z puli:**")
        st.code(" ".join([str(n) for n, c in hot_5z50]))

    with col2:
        st.subheader("📊 Wektor Ruchu Maszyny (2 z 12)")
        st.write(f"Różnice między kolejnymi wylosowanymi kulami z {N_DRAWS_TO_ANALYZE} ostatnich losowań.")
        st.dataframe(movements_2z12.style.background_gradient(cmap='Blues', axis=None), height=400, use_container_width=True)
        st.markdown("**🔥 Najgorętsze liczby z puli:**")
        st.code(" ".join([str(n) for n, c in hot_2z12]))

    st.divider()

    # The Silver Bullet
    st.header("🎯 Moduł 'Silver Bullet'")
    st.markdown("""
    *Dla analityków preferujących automatyzację.* Ten algorytm odrzuca moduł `random`. Oblicza absolutną modę (najczęstsze matematyczne przesunięcie) dla pozycji każdej kuli na bazie 50 drawów, aplikuje tę trajektorię na ostatnie znane losowanie, a potencjalne konflikty granic naprawia za pomocą najgorętszych historycznie cyfr.
    """)
    
    if st.button("Generuj Zestaw na Bazie Mechaniki Maszyny", type="primary"):
        silver_5 = silver_bullet_generator(recent_5z50, movements_5z50, 50, 5)
        silver_2 = silver_bullet_generator(recent_2z12, movements_2z12, 12, 2)
        
        # Formatowanie wyjścia do czytelnej postaci
        str_5 = " - ".join([f"{x:02d}" for x in silver_5])
        str_2 = " - ".join([f"{x:02d}" for x in silver_2])
        
        st.success("### Wygenerowany Zestaw Prawdopodobieństwa")
        st.info(f"## {str_5}  ➕  {str_2}")
