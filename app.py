import streamlit as st
import pandas as pd
import PyPDF2
import re
import numpy as np
from collections import Counter
import plotly.express as px

# ==========================================
# KONFIGURACJA GŁÓWNA
# ==========================================
st.set_page_config(page_title="Silnik Analityczny Eurojackpot", page_icon="⚙️", layout="wide")

# Ustawienie na sztywno: analizujemy 50 ostatnich losowań
N_DRAWS_TO_ANALYZE = 50 

# ==========================================
# MODUŁ EKSTRAKCJI DANYCH (PARSER BUFOROWY - SPRAWDZONY RDZEŃ)
# ==========================================
@st.cache_data
def extract_draw_data(pdf_path, expected_balls, max_ball_val):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])

        tokens = re.findall(r'\b\d+\b', text)
        
        pending_draws = []
        collected_balls = []
        data = []
        
        for t in tokens:
            val = int(t)
            is_draw_id = len(t) == 4 and (t.startswith('0') or val < 1500)
            
            if is_draw_id:
                pending_draws.append(val)
            elif 1 <= val <= max_ball_val:
                if pending_draws:
                    collected_balls.append(val)
                    if len(collected_balls) == len(pending_draws) * expected_balls:
                        for i, draw_id in enumerate(pending_draws):
                            start_idx = i * expected_balls
                            end_idx = start_idx + expected_balls
                            draw_balls = collected_balls[start_idx:end_idx]
                            
                            unique_balls = sorted(list(set(draw_balls)))
                            if len(unique_balls) == expected_balls:
                                data.append([draw_id] + unique_balls)
                        
                        pending_draws = []
                        collected_balls = []
                        
        if not data:
            return pd.DataFrame()
            
        cols = ['Losowanie'] + [f'Kula_{i+1}' for i in range(expected_balls)]
        df = pd.DataFrame(data, columns=cols)
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
    recent_df = df.head(N_DRAWS_TO_ANALYZE).copy()
    movements = recent_df.set_index('Losowanie').diff(periods=-1).dropna().astype(int)
    all_balls = recent_df.iloc[:, 1:].values.flatten()
    hot_digits = Counter(all_balls).most_common()
    
    # Dodatkowe wyliczenie odchylenia standardowego (wariancji/chaosu) dla wektorów
    variance_stats = []
    for col in movements.columns:
        mode_val = int(movements[col].mode().iloc[0])
        std_dev = movements[col].std()
        variance_stats.append({'Pozycja': col, 'Moda (Skok)': mode_val, 'Odchylenie (Chaos)': round(std_dev, 2)})
    variance_df = pd.DataFrame(variance_stats)
    
    return recent_df, movements, hot_digits, variance_df

def silver_bullet_generator(recent_df, movements, hot_digits_list, pool_size, expected_balls):
    if recent_df.empty or movements.empty:
        return []

    last_draw = recent_df.iloc[0, 1:].values
    
    likely_movements = []
    for col in movements.columns:
        mode_val = int(movements[col].mode().iloc[0])
        likely_movements.append(mode_val)
    
    generated_set = set()
    
    for ball, delta in zip(last_draw, likely_movements):
        proposed_ball = ball + delta
        if 1 <= proposed_ball <= pool_size and proposed_ball not in generated_set:
            generated_set.add(proposed_ball)
        else:
            alternative_ball = ball - delta
            if 1 <= alternative_ball <= pool_size and alternative_ball not in generated_set:
                generated_set.add(alternative_ball)
                
    if len(generated_set) < expected_balls:
        for num, _ in hot_digits_list:
            if num not in generated_set:
                generated_set.add(num)
            if len(generated_set) == expected_balls:
                break
                
    return sorted(list(generated_set))

# ==========================================
# MODUŁ WIZUALIZACJI (PLOTLY)
# ==========================================
def plot_time_river(recent_df, title):
    melted = recent_df.melt(id_vars=['Losowanie'], var_name='Pozycja', value_name='Wartość Kuli')
    melted = melted.sort_values(by='Losowanie', ascending=True)
    fig = px.line(melted, x='Losowanie', y='Wartość Kuli', color='Pozycja', markers=True, title=title, template='plotly_dark')
    fig.update_layout(xaxis=dict(type='category'))
    return fig

# ==========================================
# INTERFEJS UŻYTKOWNIKA (GŁÓWNA FUNKCJA)
# ==========================================
def main():
    st.title("🎯 Eurojackpot: Zaawansowany Silnik Analityczny")
    st.markdown("Zaprojektowane by **A.K.** | Moduł Analizy Wektorowej i Mechaniki Losowań")

    with st.spinner("Inicjalizacja parsera tokenów... Wczytywanie map Multipasko..."):
        df_5z50 = extract_draw_data("5z50.PDF", 5, 50)
        df_2z12 = extract_draw_data("2z12.PDF", 2, 12)

    if df_5z50.empty or df_2z12.empty:
        st.error("Brak poprawnych danych. Upewnij się, że pliki 5z50.PDF i 2z12.PDF są w folderze aplikacji.")
        return # Przerywa działanie funkcji głównej jeśli brak danych

    last_draw_id = df_5z50['Losowanie'].iloc[0]
    st.success(f"Analiza zakończona sukcesem. Zdekodowano historię. Ostatnie zmapowane losowanie: **{last_draw_id}**")
    
    # Procesowanie
    recent_5z50, movements_5z50, hot_5z50, var_5 = analyze_statistics(df_5z50)
    recent_2z12, movements_2z12, hot_2z12, var_2 = analyze_statistics(df_2z12)

    # --- ZAKŁADKI UI ---
    tab1, tab2, tab3 = st.tabs(["🚀 Silver Bullet", "📊 Analiza Skoków Maszyny", "🌊 Rzeka Czasu (Trendy)"])

    with tab1:
        st.header("⚡ Funkcja 'Silver Bullet'")
        st.markdown("""
        To nie jest ślepy generator `random`. Algorytm wylicza **modę skoku** (najczęściej występującą matematyczną różnicę pomiędzy poprzednim a aktualnym losem) oddzielnie dla każdej z wylosowanych pozycji w oparciu o 50 ostatnich gier. 
        Następnie nakłada te wektory na absolutnie ostatnie losowanie.
        """)
        
        if st.button("🚀 Uruchom Analizę i Generuj Zestaw", type="primary", use_container_width=True):
            silver_5 = silver_bullet_generator(recent_5z50, movements_5z50, hot_5z50, 50, 5)
            silver_2 = silver_bullet_generator(recent_2z12, movements_2z12, hot_2z12, 12, 2)
            
            str_5 = " 🟢 ".join([f"{x:02d}" for x in silver_5])
            str_2 = " 🟡 ".join([f"{x:02d}" for x in silver_2])
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("### Wytypowany zestaw na podstawie mechaniki i statystyki:")
            st.success(f"## {str_5}  ➕  {str_2}")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Bęben 1: Statystyki 5/50")
            st.dataframe(var_5.style.background_gradient(cmap='RdYlGn_r', subset=['Odchylenie (Chaos)']), use_container_width=True)
            st.markdown("**Top 10 Gorących Liczb:**")
            st.code(" | ".join([f"{n} ({c}x)" for n, c in hot_5z50[:10]]))

        with col2:
            st.subheader("Bęben 2: Statystyki 2/12")
            st.dataframe(var_2.style.background_gradient(cmap='RdYlGn_r', subset=['Odchylenie (Chaos)']), use_container_width=True)
            st.markdown("**Top 5 Gorących Liczb:**")
            st.code(" | ".join([f"{n} ({c}x)" for n, c in hot_2z12[:5]]))

    with tab3:
        st.plotly_chart(plot_time_river(recent_5z50, "Wizualizacja wektorów 5/50"), use_container_width=True)
        st.plotly_chart(plot_time_river(recent_2z12, "Wizualizacja wektorów 2/12"), use_container_width=True)


if __name__ == "__main__":
    main()
