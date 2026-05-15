import streamlit as st
import pandas as pd
import pdfplumber
import re
import numpy as np
from collections import Counter

# --- Configuration ---
st.set_page_config(page_title="Eurojackpot Analysis Engine", layout="wide")

# The engine is locked to analyze exactly the last 50 draws to track recent machine movement
N_DRAWS_TO_ANALYZE = 50 

# --- Data Extraction Modules ---
@st.cache_data
def extract_draw_data(pdf_path, expected_balls):
    """
    Extracts draw numbers and ball results from the specific multipasko PDF format.
    """
    raw_data = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text:
                    continue
                
                # Regex to find draw numbers (e.g., 0954) followed by potential ball numbers
                lines = text.split('\n')
                for line in lines:
                    # Look for lines that start with a 4-digit draw number
                    match = re.match(r'^(\d{4})\s+(.*)', line.strip())
                    if match:
                        draw_num = int(match.group(1))
                        # Extract all 2-digit numbers from the rest of the line
                        balls = [int(b) for b in re.findall(r'\b\d{2}\b', match.group(2))]
                        
                        if len(balls) == expected_balls:
                            raw_data.append([draw_num] + balls)
                            
        if not raw_data:
            return pd.DataFrame()
            
        # Sort by draw number descending (newest first)
        df = pd.DataFrame(raw_data).sort_values(by=0, ascending=False).reset_index(drop=True)
        # Rename columns
        cols = ['Draw'] + [f'Ball_{i+1}' for i in range(expected_balls)]
        df.columns = cols
        return df
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {e}")
        return pd.DataFrame()

# --- Analytical Core ---
def analyze_machine_movement(df):
    """
    Analyzes the mathematical pattern of 'machine movement' (difference between consecutive draws) 
    strictly over the last 50 draws.
    """
    # Isolate the exact historical window
    recent_df = df.head(N_DRAWS_TO_ANALYZE).copy()
    
    # Calculate differences (deltas) between row i and row i+1
    # shift(-1) moves the older draw up to compare with the newer draw
    movements = recent_df.set_index('Draw').diff(periods=-1).dropna().abs()
    
    return recent_df, movements

def get_hot_digits(df, num_balls):
    """Calculates the most frequent balls in the analyzed dataset."""
    all_balls = df.iloc[:, 1:].values.flatten()
    counter = Counter(all_balls)
    return counter.most_common(num_balls)

def silver_bullet_generator(recent_df, movements, pool_size, expected_balls):
    """
    The Silver Bullet: Generates a likely set based on the most frequent
    machine movements applied to the very last draw, prioritizing hot numbers.
    """
    if recent_df.empty or movements.empty:
        return []

    last_draw = recent_df.iloc[0, 1:].values
    
    # Find the most common movement (delta) for each ball position
    likely_movements = []
    for col in movements.columns:
        # Get the mode (most frequent delta). If multiple, take the first.
        most_common_delta = int(movements[col].mode()[0])
        likely_movements.append(most_common_delta)
    
    generated_set = set()
    
    # Apply movement to the last draw
    for i, (ball, delta) in enumerate(zip(last_draw, likely_movements)):
        # Machine movement can go up or down; we pick the direction 
        # that keeps it in bounds and hasn't been used yet.
        option_up = ball + delta
        option_down = ball - delta
        
        if 1 <= option_up <= pool_size and option_up not in generated_set:
            generated_set.add(option_up)
        elif 1 <= option_down <= pool_size and option_down not in generated_set:
            generated_set.add(option_down)
            
    # If the mechanical movement didn't yield enough unique valid balls, 
    # fill the rest with the hottest numbers from the 50-draw pool.
    if len(generated_set) < expected_balls:
        hot_digits = get_hot_digits(recent_df, expected_balls * 2)
        for num, _ in hot_digits:
            if num not in generated_set:
                generated_set.add(num)
            if len(generated_set) == expected_balls:
                break
                
    return sorted(list(generated_set))

# --- Streamlit UI ---
st.title("⚙️ Eurojackpot Analytical Engine")
st.markdown("Joint Collaboration: 30-Year Code Veteran & AI Analytical Core")

# Load Data
st.sidebar.header("Data Ingestion")
st.sidebar.info("Looking for `5z50.PDF` and `2z12.PDF` in the root directory.")

with st.spinner("Parsing GitHub PDF Directories..."):
    df_5z50 = extract_draw_data("5z50.PDF", 5)
    df_2z12 = extract_draw_data("2z12.PDF", 2)

if df_5z50.empty or df_2z12.empty:
    st.error("Engine halted. Ensure '5z50.PDF' and '2z12.PDF' are in the root directory and structured correctly. Check that the PDF text layer contains 4-digit draw numbers followed by 2-digit ball numbers.")
else:
    st.success("Data successfully ingested and parsed.")
    
    # Process 5z50
    recent_5z50, movements_5z50 = analyze_machine_movement(df_5z50)
    hot_5z50 = get_hot_digits(recent_5z50, 10)
    
    # Process 2z12
    recent_2z12, movements_2z12 = analyze_machine_movement(df_2z12)
    hot_2z12 = get_hot_digits(recent_2z12, 5)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 5/50 Machine Movement Analysis")
        st.write(f"Tracking positional momentum across the last {N_DRAWS_TO_ANALYZE} draws.")
        st.dataframe(movements_5z50.style.background_gradient(cmap='Greens'), height=400)
        st.write("**Top Hot Digits (Last 50 Draws):**")
        st.write([n for n, c in hot_5z50])

    with col2:
        st.subheader("📊 2/12 Machine Movement Analysis")
        st.write(f"Tracking positional momentum across the last {N_DRAWS_TO_ANALYZE} draws.")
        st.dataframe(movements_2z12.style.background_gradient(cmap='Blues'), height=400)
        st.write("**Top Hot Digits (Last 50 Draws):**")
        st.write([n for n, c in hot_2z12])

    st.divider()

    # The Silver Bullet
    st.header("🎯 The Silver Bullet")
    st.markdown("""
    *For the lazy, but mathematically inclined.* This generator bypasses random number generation. It calculates the exact modal delta (most frequent mechanical jump) for each ball position over the last 50 draws, applies it to the very last drawn set, and cross-references missing values with the hottest historical digits.
    """)
    
    if st.button("Generate Set via Machine Movement", type="primary"):
        silver_5 = silver_bullet_generator(recent_5z50, movements_5z50, 50, 5)
        silver_2 = silver_bullet_generator(recent_2z12, movements_2z12, 12, 2)
        
        st.success(f"### Proposed Set: {silver_5} + {silver_2}")
