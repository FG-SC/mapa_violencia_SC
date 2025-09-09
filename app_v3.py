import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import geopandas as gpd
import json
from datetime import datetime
import warnings
import os
from scipy import stats
from io import BytesIO
import folium
from folium import plugins
from folium.plugins import HeatMap
import streamlit.components.v1 as components

warnings.filterwarnings('ignore')

# Import config utilities with fallback
try:
    from config_utils import (
        SC_MESOREGIONS_DETAILED, 
        DataProcessor, 
        VisualizationHelper, 
        StatisticalAnalyzer,
        assess_data_quality,
        STREAMLIT_CONFIG
    )
except ImportError:
    st.warning("⚠️ config_utils.py não encontrado. Usando configurações padrão.")
    
    # Simplified fallback mesoregions
    SC_MESOREGIONS_DETAILED = {
        'Oeste Catarinense': ['Chapecó', 'São Miguel do Oeste', 'Xanxerê', 'Concórdia'],
        'Norte Catarinense': ['Joinville', 'São Bento do Sul', 'Mafra', 'Canoinhas'],
        'Vale do Itajaí': ['Blumenau', 'Pomerode', 'Gaspar', 'Indaial', 'Timbó'],
        'Grande Florianópolis': ['Florianópolis', 'São José', 'Palhoça', 'Biguaçu'],
        'Sul Catarinense': ['Criciúma', 'Tubarão', 'Araranguá', 'Laguna'],
        'Serrana': ['Lages', 'São Joaquim', 'Campos Novos', 'Curitibanos']
    }
    
    class DataProcessor:
        @staticmethod
        def get_mesoregion(municipality):
            if pd.isna(municipality):
                return 'Não Informado'
            municipality_clean = str(municipality).strip().title()
            for region, cities in SC_MESOREGIONS_DETAILED.items():
                for city in cities:
                    if city.lower() in municipality_clean.lower():
                        return region
            return 'Outras Regiões'

# Page configuration
st.set_page_config(
    page_title="🔍 Análise de Violência contra a Mulher - SC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for page persistence
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 0
if 'selected_crime' not in st.session_state:
    st.session_state.selected_crime = 'Todos os Crimes'

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        border-left: 4px solid #FF6B6B;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #8B4513;
        border-left: 4px solid #FF8C00;
    }
    .success-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: #2F4F4F;
        border-left: 4px solid #4ECDC4;
    }
    /* Multi-index table styling */
    .dataframe th {
        background-color: #f0f2f6 !important;
        font-weight: bold !important;
    }
    .dataframe tbody tr th:first-child {
        background-color: #e0e3e9 !important;
        font-weight: bold !important;
    }
    /* Style for multi-index category cells */
    .dataframe tbody tr th.level0 {
        background-color: #d4d7dd !important;
        font-weight: bold !important;
        text-align: left !important;
        padding-left: 10px !important;
    }
</style>
""", unsafe_allow_html=True)


# Improved education-crime correlation analysis
def create_education_crime_correlation(df):
    """Analyze correlation between education level and crime occurrences with better interpretation"""
    if 'Escolaridade - Agressor' not in df.columns or 'Tipo penal' not in df.columns:
        return None, None
    
    # Define education level ordering (lower number = lower education)
    education_levels = {
        'Não Alfabetizado': 1,
        'Alfabetizado': 2,
        'Fundamental Incompleto': 3,
        'Fundamental Completo': 4,
        'Médio Incompleto': 5,
        'Médio Completo': 6,
        'Superior Incompleto': 7,
        'Superior Completo': 8,
        'Pós-graduação': 9
    }
    
    # Clean data
    df_clean = df.dropna(subset=['Escolaridade - Agressor'])
    df_clean = df_clean[df_clean['Escolaridade - Agressor'] != 'Não Informado']
    
    # Map education to numeric levels
    df_clean['Education_Level'] = df_clean['Escolaridade - Agressor'].map(education_levels)
    df_clean = df_clean.dropna(subset=['Education_Level'])
    
    # Count crimes by education level and type
    crime_counts = df_clean.groupby(['Education_Level', 'Escolaridade - Agressor', 'Tipo penal']).size().reset_index(name='Crime_Count')
    
    # Calculate correlation
    if len(crime_counts) > 1:
        correlation, p_value = stats.pearsonr(crime_counts['Education_Level'], crime_counts['Crime_Count'])
    else:
        correlation, p_value = 0, 1
    
    # Create visualization
    fig = px.scatter(
        crime_counts,
        x='Education_Level',
        y='Crime_Count',
        color='Tipo penal',
        text='Escolaridade - Agressor',
        title=f'📊 Correlação: Nível de Escolaridade vs Ocorrência de Crimes<br>Correlação de Pearson: {correlation:.3f} (p-valor: {p_value:.3f})',
        labels={'Education_Level': 'Nível de Escolaridade (1=Menor, 9=Maior)', 'Crime_Count': 'Número de Crimes'},
        trendline='ols',
        hover_data=['Tipo penal']
    )
    
    fig.update_traces(textposition='top center', marker=dict(size=12))
    fig.update_layout(
        height=600,
        legend_title_text='Tipo de Crime'
    )
    
    # Add interpretation
    interpretation = ""
    if p_value < 0.05:
        if correlation < 0:
            interpretation = "📉 **Correlação negativa significativa**: Níveis mais baixos de escolaridade estão associados a maior ocorrência de crimes."
        else:
            interpretation = "📈 **Correlação positiva significativa**: Níveis mais altos de escolaridade estão associados a maior ocorrência de crimes. Isso pode indicar que agressores com mais educação cometem crimes mais frequentemente ou que há maior denúncia desses casos."
    else:
        interpretation = "➖ **Sem correlação significativa**: Não há relação estatisticamente significativa entre nível educacional e ocorrência de crimes."
    
    return fig, {'correlation': correlation, 'p_value': p_value, 'data': crime_counts, 'interpretation': interpretation}

def get_mesoregion(municipality):
    """Map municipality to meso-region - optimized version"""
    try:
        return DataProcessor.get_mesoregion(municipality)
    except:
        # Fast fallback function
        if pd.isna(municipality):
            return 'Não Informado'
        
        municipality_lower = str(municipality).lower()
        
        # Quick lookup for major cities only
        major_cities_mapping = {
            'florianópolis': 'Grande Florianópolis',
            'florianopolis': 'Grande Florianópolis',
            'palhoça': 'Grande Florianópolis',
            'palhoca': 'Grande Florianópolis',
            'são josé': 'Grande Florianópolis',
            'sao jose': 'Grande Florianópolis',
            'biguaçu': 'Grande Florianópolis',
            'biguacu': 'Grande Florianópolis',
            'joinville': 'Norte Catarinense',
            'blumenau': 'Vale do Itajaí',
            'chapecó': 'Oeste Catarinense',
            'chapeco': 'Oeste Catarinense',
            'criciúma': 'Sul Catarinense',
            'criciuma': 'Sul Catarinense',
            'itajaí': 'Vale do Itajaí',
            'itajai': 'Vale do Itajaí',
            'lages': 'Serrana',
            'tubarão': 'Sul Catarinense',
            'tubarao': 'Sul Catarinense',
            'são bento do sul': 'Norte Catarinense',
            'sao bento do sul': 'Norte Catarinense'
        }
        
        for city, region in major_cities_mapping.items():
            if city in municipality_lower:
                return region
        
        return 'Outras Regiões'

def create_multi_index_summary_table(df):
    """Create comprehensive multi-index summary table like the printed image"""
    if len(df) == 0:
        return None, None
    
    # Define crime categories based on Categorias.xlsx
    crime_categories = {
        'Violência Física': ['Lesão Corporal', 'Homicídio', 'Agressão', 'Tentativa De Homicídio'],
        'Violência Psicológica': ['Ameaça', 'Perseguição', 'Constrangimento Ilegal', 'Perturbação Da Tranquilidade'],
        'Violência Moral': ['Injúria', 'Difamação', 'Calúnia'],
        'Violência Sexual': ['Estupro', 'Importunação Sexual', 'Assédio Sexual', 'Violação Sexual'],
        'Violência Econômica/Patrimonial': ['Dano', 'Furto', 'Roubo', 'Apropriação Indébita', 'Estelionato'],
        'Feminicídio': ['Feminicídio', 'Tentativa De Feminicídio'],
        'Múltiplas Dimensões (Outros)': []  # Will include remaining crimes
    }
    
    # Get all unique crimes in the dataset
    all_crimes = df['Tipo penal'].dropna().unique().tolist()
    
    # Assign remaining crimes to 'Múltiplas Dimensões (Outros)'
    assigned_crimes = []
    for crimes in crime_categories.values():
        assigned_crimes.extend(crimes)
    
    crime_categories['Múltiplas Dimensões (Outros)'] = [crime for crime in all_crimes if crime not in assigned_crimes and crime != 'Não Informado']
    
    # Function to create age groups
    def create_age_groups(age_series):
        if age_series.isna().all():
            return pd.Series(['Não Informado'] * len(age_series), index=age_series.index)
        
        age_groups = pd.cut(age_series, 
                     bins=[0, 4, 9, 14, 19, 29, 39, 49, 59, 69, 100],
                     labels=['0 a 4 anos', '5 a 9 anos', '10 a 14 anos', '15 a 19 anos',
                            '20 a 29 anos', '30 a 39 anos', '40 a 49 anos', '50 a 59 anos',
                            '60 a 69 anos', '70 anos ou mais'],
                     include_lowest=True)
        
        age_groups_str = age_groups.astype(str).replace('nan', 'Não Informado')
        return age_groups_str
    
    # Prepare the summary sections
    all_sections = []
    
    # Map categories to process
    categories_to_process = [
        ('Faixa etária', 'idade - mulher', create_age_groups),
        ('Vínculo', 'Vínculo', None),
        ('Escolaridade', 'Escolaridade - mulher', None),
        ('Regiões', 'Mesorregião', None)
    ]
    
    for category_name, col_name, transform_func in categories_to_process:
        if col_name not in df.columns:
            continue
            
        df_temp = df.copy()
        
        # Apply transformation if needed
        if transform_func:
            df_temp['_category'] = transform_func(df_temp[col_name])
        else:
            df_temp['_category'] = df_temp[col_name]
        
        # Remove invalid entries
        df_temp = df_temp.dropna(subset=['_category'])
        df_temp = df_temp[df_temp['_category'] != 'Não Informado']
        
        # Get unique values
        unique_values = df_temp['_category'].value_counts().index.tolist()
        if category_name == 'Vínculo':
            unique_values = unique_values[:10]  # Top 10 relationships
        elif category_name == 'Escolaridade':
            unique_values = unique_values[:8]  # Top 8 education levels
        
        # Process each subcategory
        for subcategory in unique_values:
            subcategory_data = df_temp[df_temp['_category'] == subcategory]
            
            if len(subcategory_data) == 0:
                continue
            
            row_data = {}
            total_cases = len(subcategory_data)
            
            # Calculate for each crime category
            for crime_cat, crimes in crime_categories.items():
                if crimes:
                    crime_cases = len(subcategory_data[subcategory_data['Tipo penal'].isin(crimes)])
                    row_data[f'{crime_cat}_N'] = crime_cases
                    row_data[f'{crime_cat}_%'] = round((crime_cases / total_cases * 100), 1) if total_cases > 0 else 0.0
            
            row_data['Total_N'] = total_cases
            row_data['Total_%'] = 100.0
            
            all_sections.append({
                'Categoria': category_name,
                'Subcategoria': str(subcategory),
                **row_data
            })
    
    if not all_sections:
        return None, None
    
    # Create DataFrame
    summary_df = pd.DataFrame(all_sections)
    
    # Set up multi-index
    summary_df_indexed = summary_df.set_index(['Categoria', 'Subcategoria'])
    
    # Create column structure
    column_data = []
    column_names = []
    
    for crime_cat in crime_categories.keys():
        if f'{crime_cat}_N' in summary_df.columns:
            column_data.extend([f'{crime_cat}_N', f'{crime_cat}_%'])
            column_names.extend([(crime_cat, 'N'), (crime_cat, '%')])
    
    column_data.extend(['Total_N', 'Total_%'])
    column_names.extend([('Total', 'N'), ('Total', '%')])
    
    # Reorder and rename columns
    summary_df_final = summary_df_indexed[column_data]
    summary_df_final.columns = pd.MultiIndex.from_tuples(column_names)
    
    # Convert N columns to integers
    for col in summary_df_final.columns:
        if col[1] == 'N':
            summary_df_final[col] = summary_df_final[col].fillna(0).astype(int)
    
    return summary_df, summary_df_final

def create_mesoregion_choropleth(df, selected_crime='Todos os Crimes'):
    """Create choropleth map by mesoregions"""
    # Filter by crime type
    if selected_crime != 'Todos os Crimes':
        df_filtered = df[df['Tipo penal'] == selected_crime]
    else:
        df_filtered = df.copy()
    
    # Aggregate by mesoregion
    if 'Mesorregião' not in df_filtered.columns:
        return None
    
    meso_counts = df_filtered['Mesorregião'].value_counts().reset_index()
    meso_counts.columns = ['Mesorregião', 'Casos']
    
    # Define mesoregion colors based on the reference map
    mesoregion_colors = {
        'Oeste Catarinense': '#FF99FF',  # Pink/Magenta
        'Norte Catarinense': '#FFB3D9',  # Light Pink
        'Vale do Itajaí': '#FFFF99',     # Yellow
        'Serrana': '#FFD4AA',            # Light Orange
        'Grande Florianópolis': '#99FF99', # Light Green
        'Sul Catarinense': '#B3B3FF',     # Light Blue
        'Outras Regiões': '#E0E0E0'       # Gray
    }
    
    # Create plotly figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=meso_counts['Mesorregião'],
        y=meso_counts['Casos'],
        marker=dict(
            color=[mesoregion_colors.get(r, '#E0E0E0') for r in meso_counts['Mesorregião']],
            line=dict(color='black', width=1.5)
        ),
        text=meso_counts['Casos'],
        textposition='auto',
        hovertemplate='%{x}<br>Casos: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'🗺️ Casos por Mesorregião - {selected_crime}',
        xaxis_title='Mesorregião',
        yaxis_title='Número de Casos',
        template='plotly_white',
        height=500,
        showlegend=False
    )
    
    return fig, meso_counts

def standardize_education(education):
    """Standardize education levels with aggregation"""
    if pd.isna(education) or education in ['Não Informado', '', 'nan']:
        return 'Não Informado'
    
    education_clean = str(education).strip().lower()
    
    # Aggregate semi-literate with non-literate
    if any(term in education_clean for term in ['semianalfabetizado', 'semi-alfabetizado', 'semi alfabetizado']):
        return 'Não Alfabetizado'
    
    # Aggregate all post-graduation
    if any(term in education_clean for term in ['pós', 'pos', 'mestrado', 'doutorado', 'especialização']):
        return 'Pós-graduação'
    
    # Aggregate graduação cursando with superior incompleto
    if 'graduação' in education_clean and 'cursando' in education_clean:
        return 'Superior Incompleto'
    
    # Map other education levels
    education_mapping = {
        'analfabeto': 'Não Alfabetizado',
        'não alfabetizado': 'Não Alfabetizado',
        'não alfabetizada': 'Não Alfabetizado',
        'alfabetizado': 'Alfabetizado',
        'ensino fundamental incompleto': 'Fundamental Incompleto',
        'fundamental incompleto': 'Fundamental Incompleto',
        'ensino fundamental completo': 'Fundamental Completo',
        'fundamental completo': 'Fundamental Completo',
        'ensino médio incompleto': 'Médio Incompleto',
        'médio incompleto': 'Médio Incompleto',
        'ensino médio completo': 'Médio Completo',
        'médio completo': 'Médio Completo',
        'superior incompleto': 'Superior Incompleto',
        'ensino superior incompleto': 'Superior Incompleto',
        'superior completo': 'Superior Completo',
        'ensino superior completo': 'Superior Completo'
    }
    
    for key, value in education_mapping.items():
        if key in education_clean:
            return value
    
    return education_clean.title()

def create_race_analysis(df):
    """Create race/ethnicity analysis from q26 responses"""
    race_columns = ['q26_0', 'q26_1', 'q26_2', 'q26_3', 'q26_4']
    race_labels = {
        'q26_0': 'Branca',
        'q26_1': 'Preta', 
        'q26_2': 'Parda',
        'q26_3': 'Amarela',
        'q26_4': 'Indígena'
    }
    
    # Check if race columns exist
    existing_cols = [col for col in race_columns if col in df.columns]
    if not existing_cols:
        return None, None
    
    # Calculate race distribution
    race_counts = {}
    for col in existing_cols:
        if col in df.columns:
            count = df[col].sum() if df[col].dtype in ['int64', 'float64'] else len(df[df[col] == 1])
            race_counts[race_labels[col]] = count
    
    # Create DataFrame
    race_df = pd.DataFrame(list(race_counts.items()), columns=['Raça/Cor', 'Quantidade'])
    race_df['Porcentagem'] = (race_df['Quantidade'] / race_df['Quantidade'].sum() * 100).round(2)
    race_df = race_df.sort_values('Quantidade', ascending=False)
    
    # Create visualization
    fig = px.bar(
        race_df,
        x='Raça/Cor',
        y='Quantidade',
        title='📊 Distribuição por Raça/Cor das Vítimas',
        color='Raça/Cor',
        color_discrete_map={
            'Branca': '#3498db',
            'Preta': '#e74c3c',
            'Parda': '#f39c12',
            'Amarela': '#f1c40f',
            'Indígena': '#27ae60'
        },
        text='Quantidade'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(height=500, showlegend=False)
    
    return fig, race_df

def create_questionnaire_response_table(df):
    """Create multi-index table for questionnaire responses"""
    # Get all question columns
    q_columns = [col for col in df.columns if col.startswith('q') and '_' in col and not col.startswith('q26')]
    
    if not q_columns:
        return None
    
    # Create response summary
    response_data = []
    
    for col in sorted(q_columns):
        if col in df.columns:
            # Count responses
            if df[col].dtype in ['int64', 'float64', 'bool']:
                # For numeric columns, sum the values (assuming 1 = yes, 0 = no)
                count = int(df[col].sum())
            else:
                # For other types, count non-null values
                count = int(df[col].notna().sum())
            
            total = len(df)
            percentage = (count / total * 100) if total > 0 else 0
            
            # Extract question and option number
            parts = col.split('_')
            if len(parts) >= 2:
                question = parts[0]
                option = parts[1]
                
                response_data.append({
                    'Pergunta': question,
                    'Alternativa': option,
                    'N': count,
                    '%': round(percentage, 1)
                })
    
    if not response_data:
        return None
    
    # Create DataFrame
    response_df = pd.DataFrame(response_data)
    
    # Create multi-index
    response_df_indexed = response_df.set_index(['Pergunta', 'Alternativa'])
    
    return response_df_indexed

def analyze_psychological_indicators(df):
    """Analyze q34 and q35 for psychological indicators"""
    results = {}
    
    # Analyze q34 - Physical and emotional presentation
    if 'q34' in df.columns:
        q34_data = df['q34'].value_counts()
        
        fig_q34 = px.bar(
            x=q34_data.index[:10],  # Top 10 responses
            y=q34_data.values[:10],
            title='📊 Q34: Apresentação Física e Emocional das Vítimas',
            labels={'x': 'Resposta', 'y': 'Frequência'},
            color_discrete_sequence=['#e74c3c']
        )
        fig_q34.update_xaxes(tickangle=45)
        fig_q34.update_layout(height=500)
        
        results['q34'] = {'fig': fig_q34, 'data': q34_data}
    
    # Analyze q35 - Suicide risk
    if 'q35' in df.columns:
        q35_data = df['q35'].value_counts()
        
        fig_q35 = px.bar(
            x=q35_data.index,
            y=q35_data.values,
            title='📊 Q35: Risco de Tentativa de Suicídio',
            labels={'x': 'Resposta', 'y': 'Frequência'},
            color_discrete_sequence=['#c0392b']
        )
        fig_q35.update_layout(height=400)
        
        results['q35'] = {'fig': fig_q35, 'data': q35_data}
    
    return results

def create_risk_score_analysis(df):
    """Create risk score based on multiple indicators"""
    risk_factors = {
        'q1_0': 3,  # Threat with firearm
        'q1_1': 2,  # Threat with knife
        'q2_4': 3,  # Shot
        'q2_6': 3,  # Stabbed
        'q35': 5    # Suicide risk
    }
    
    # Calculate risk score for each case
    df['risk_score'] = 0
    
    for factor, weight in risk_factors.items():
        if factor in df.columns:
            if df[factor].dtype in ['int64', 'float64']:
                df['risk_score'] += df[factor] * weight
            else:
                df['risk_score'] += (df[factor] == 1).astype(int) * weight
    
    # Categorize risk levels
    df['risk_level'] = pd.cut(
        df['risk_score'],
        bins=[-1, 2, 5, 10, 100],
        labels=['Baixo', 'Médio', 'Alto', 'Extremo']
    )
    
    # Create visualization
    risk_counts = df['risk_level'].value_counts()
    
    fig = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title='📊 Distribuição por Nível de Risco',
        color_discrete_map={
            'Baixo': '#27ae60',
            'Médio': '#f39c12',
            'Alto': '#e67e22',
            'Extremo': '#c0392b'
        }
    )
    
    return fig, df[['risk_score', 'risk_level']]
    """Analyze correlation between education level and crime occurrences"""
    if 'Escolaridade - Agressor' not in df.columns or 'Tipo penal' not in df.columns:
        return None, None
    
    # Define education level ordering (lower number = lower education)
    education_levels = {
        'Não Alfabetizado': 1,
        'Alfabetizado': 2,
        'Fundamental Incompleto': 3,
        'Fundamental Completo': 4,
        'Médio Incompleto': 5,
        'Médio Completo': 6,
        'Superior Incompleto': 7,
        'Superior Completo': 8,
        'Pós-graduação': 9
    }
    
    # Clean data
    df_clean = df.dropna(subset=['Escolaridade - Agressor'])
    df_clean = df_clean[df_clean['Escolaridade - Agressor'] != 'Não Informado']
    
    # Map education to numeric levels
    df_clean['Education_Level'] = df_clean['Escolaridade - Agressor'].map(education_levels)
    df_clean = df_clean.dropna(subset=['Education_Level'])
    
    # Count crimes by education level
    crime_counts = df_clean.groupby(['Education_Level', 'Escolaridade - Agressor']).size().reset_index(name='Crime_Count')
    
    # Calculate correlation
    if len(crime_counts) > 1:
        correlation, p_value = stats.pearsonr(crime_counts['Education_Level'], crime_counts['Crime_Count'])
    else:
        correlation, p_value = 0, 1
    
    # Create visualization
    fig = px.scatter(
        crime_counts,
        x='Education_Level',
        y='Crime_Count',
        text='Escolaridade - Agressor',
        title=f'📊 Correlação: Nível de Escolaridade vs Ocorrência de Crimes<br>Correlação de Pearson: {correlation:.3f} (p-valor: {p_value:.3f})',
        labels={'Education_Level': 'Nível de Escolaridade (1=Menor, 9=Maior)', 'Crime_Count': 'Número de Crimes'},
        trendline='ols'
    )
    
    fig.update_traces(textposition='top center', marker=dict(size=12, color='red'))
    fig.update_layout(height=600)
    
    return fig, {'correlation': correlation, 'p_value': p_value, 'data': crime_counts}

def create_education_correlation_heatmap(df):
    """Create heatmap showing correlation between victim and aggressor education levels"""
    if 'Escolaridade - mulher' not in df.columns or 'Escolaridade - Agressor' not in df.columns:
        return None
    
    # Clean data
    df_clean = df.dropna(subset=['Escolaridade - mulher', 'Escolaridade - Agressor'])
    df_clean = df_clean[
        (df_clean['Escolaridade - mulher'] != 'Não Informado') & 
        (df_clean['Escolaridade - Agressor'] != 'Não Informado')
    ]
    
    if len(df_clean) == 0:
        return None
    
    # Create cross-tabulation
    crosstab = pd.crosstab(
        df_clean['Escolaridade - mulher'], 
        df_clean['Escolaridade - Agressor']
    )
    
    # Order education levels
    education_order = [
        'Não Alfabetizado', 'Alfabetizado', 'Fundamental Incompleto',
        'Fundamental Completo', 'Médio Incompleto', 'Médio Completo',
        'Superior Incompleto', 'Superior Completo', 'Pós-graduação'
    ]
    
    # Reorder if possible
    available_rows = [e for e in education_order if e in crosstab.index]
    available_cols = [e for e in education_order if e in crosstab.columns]
    crosstab = crosstab.reindex(index=available_rows, columns=available_cols, fill_value=0)
    
    # Create heatmap
    fig = px.imshow(
        crosstab.values,
        x=crosstab.columns,
        y=crosstab.index,
        title='🔥 Mapa de Calor: Escolaridade Vítima vs Agressor',
        labels=dict(x="Escolaridade do Agressor", y="Escolaridade da Vítima", color="Número de Casos"),
        color_continuous_scale='YlOrRd',
        aspect='auto',
        text_auto=True
    )
    
    fig.update_layout(
        height=600,
        xaxis_tickangle=45
    )
    
    return fig

def create_municipality_heatmap(df, selected_crime='Todos os Crimes'):
    """Create interactive municipality heatmap with hover"""
    if 'Município' not in df.columns:
        return None
    
    # Filter by crime type
    if selected_crime != 'Todos os Crimes':
        df_filtered = df[df['Tipo penal'] == selected_crime]
    else:
        df_filtered = df.copy()
    
    # Aggregate by municipality
    muni_counts = df_filtered['Município'].value_counts().reset_index()
    muni_counts.columns = ['Município', 'Casos']
    
    # Create base map
    center_lat, center_lon = -27.2423, -50.2189
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=7, 
        tiles='OpenStreetMap',
        control_scale=True
    )
    
    # Try to load municipality boundaries
    try:
        import geobr
        sc_municipalities = geobr.read_municipality(code_muni="SC", year=2020)
        
        # Merge with data
        sc_merged = sc_municipalities.merge(
            muni_counts,
            left_on='name_muni',
            right_on='Município',
            how='left'
        )
        sc_merged['Casos'] = sc_merged['Casos'].fillna(0)
        
        # Create choropleth
        folium.Choropleth(
            geo_data=sc_merged,
            name='choropleth',
            data=sc_merged,
            columns=['name_muni', 'Casos'],
            key_on='feature.properties.name_muni',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name=f'Número de Casos - {selected_crime}',
            nan_fill_color='white'
        ).add_to(m)
        
        # Add hover functionality
        style_function = lambda x: {
            'fillColor': '#ffffff',
            'color': '#000000',
            'fillOpacity': 0.1,
            'weight': 0.1
        }
        
        highlight_function = lambda x: {
            'fillColor': '#000000',
            'color': '#000000',
            'fillOpacity': 0.50,
            'weight': 3
        }
        
        tooltip = folium.features.GeoJson(
            sc_merged,
            style_function=style_function,
            control=False,
            highlight_function=highlight_function,
            tooltip=folium.features.GeoJsonTooltip(
                fields=['name_muni', 'Casos'],
                aliases=['Município:', 'Casos:'],
                style="background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;",
                sticky=True
            )
        )
        tooltip.add_to(m)
        
    except Exception as e:
        # Fallback to heat map with approximate locations
        from folium.plugins import HeatMap
        
        # Generate approximate coordinates (in production, use proper geocoding)
        heat_data = []
        for _, row in muni_counts.head(50).iterrows():  # Limit to top 50
            # This is simplified - in production, you'd want real coordinates
            lat = center_lat + np.random.uniform(-2, 2)
            lon = center_lon + np.random.uniform(-3, 3)
            weight = row['Casos'] / muni_counts['Casos'].max()
            heat_data.append([lat, lon, weight])
        
        # Add heatmap
        HeatMap(
            heat_data,
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={
                0.0: 'blue',
                0.2: 'cyan',
                0.4: 'green',
                0.6: 'yellow',
                0.8: 'orange',
                1.0: 'red'
            }
        ).add_to(m)
    
    # Add color scale legend
    colormap_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 250px; height: 90px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin: 0;"><b>Escala de Intensidade</b></p>
    <div style="background: linear-gradient(to right, #FFEDA0, #FED976, #FEB24C, #FD8D3C, #FC4E2A, #E31A1C, #BD0026, #800026); 
                height: 20px; margin: 5px 0;"></div>
    <p style="margin: 0; display: flex; justify-content: space-between;">
        <span>Min</span><span>Max</span>
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(colormap_html))
    
    return m

def create_seasonality_analysis(df):
    """Create seasonality analysis for temporal patterns"""
    if 'Data do registro' not in df.columns:
        return None, None
    
    df_clean = df.dropna(subset=['Data do registro'])
    if len(df_clean) == 0:
        return None, None
    
    # Extract temporal features
    df_clean = df_clean.copy()
    df_clean['Month'] = df_clean['Data do registro'].dt.month
    df_clean['Month_Name'] = df_clean['Data do registro'].dt.strftime('%B')
    df_clean['Year'] = df_clean['Data do registro'].dt.year
    
    # Monthly aggregation
    monthly_counts = df_clean.groupby('Month').size().reset_index(name='Cases')
    
    # Add month names in Portuguese
    month_names_pt = {
        1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril',
        5: 'Maio', 6: 'Junho', 7: 'Julho', 8: 'Agosto',
        9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'
    }
    monthly_counts['Month_Name'] = monthly_counts['Month'].map(month_names_pt)
    
    # Calculate statistics
    mean_cases = monthly_counts['Cases'].mean()
    std_cases = monthly_counts['Cases'].std()
    
    # Create bar chart with seasonality indicators
    fig1 = go.Figure()
    
    # Add bars
    fig1.add_trace(go.Bar(
        x=monthly_counts['Month_Name'],
        y=monthly_counts['Cases'],
        marker_color=['red' if x > mean_cases + std_cases else 'lightblue' for x in monthly_counts['Cases']],
        text=monthly_counts['Cases'],
        textposition='auto',
        name='Casos por Mês'
    ))
    
    # Add mean line
    fig1.add_hline(
        y=mean_cases, 
        line_dash="dash", 
        line_color="green",
        annotation_text=f"Média: {mean_cases:.0f}"
    )
    
    # Add upper threshold
    fig1.add_hline(
        y=mean_cases + std_cases, 
        line_dash="dot", 
        line_color="red",
        annotation_text=f"Alto: {mean_cases + std_cases:.0f}"
    )
    
    fig1.update_layout(
        title='📊 Análise de Sazonalidade - Casos por Mês',
        xaxis_title='Mês',
        yaxis_title='Número de Casos',
        height=500,
        showlegend=False
    )
    
    # Year-over-year comparison if multiple years
    if df_clean['Year'].nunique() > 1:
        yearly_monthly = df_clean.groupby(['Year', 'Month']).size().reset_index(name='Cases')
        yearly_monthly['Month_Name'] = yearly_monthly['Month'].map(month_names_pt)
        
        fig2 = px.line(
            yearly_monthly,
            x='Month_Name',
            y='Cases',
            color='Year',
            title='📈 Comparação Ano a Ano - Padrões Mensais',
            labels={'Cases': 'Número de Casos', 'Month_Name': 'Mês'},
            markers=True
        )
        
        return fig1, fig2
    
    return fig1, None

def create_crime_category_boxplot(df):
    """Create boxplot of victim age by crime categories"""
    if 'idade - mulher' not in df.columns or 'Tipo penal' not in df.columns:
        return None
    
    # Define crime categories mapping based on Categorias.xlsx
    crime_mapping = {
        'Violência Física': ['Lesão Corporal', 'Homicídio', 'Agressão', 'Tentativa De Homicídio'],
        'Violência Psicológica': ['Ameaça', 'Perseguição', 'Constrangimento Ilegal', 'Perturbação Da Tranquilidade'],
        'Violência Moral': ['Injúria', 'Difamação', 'Calúnia'],
        'Violência Sexual': ['Estupro', 'Importunação Sexual', 'Assédio Sexual', 'Violação Sexual'],
        'Violência Econômica/Patrimonial': ['Dano', 'Furto', 'Roubo', 'Apropriação Indébita', 'Estelionato'],
        'Feminicídio': ['Feminicídio', 'Tentativa De Feminicídio'],
        'Múltiplas Dimensões (Outros)': []
    }
    
    # Create reverse mapping
    crime_to_category = {}
    for category, crimes in crime_mapping.items():
        for crime in crimes:
            crime_to_category[crime] = category
    
    # Clean data
    df_clean = df.dropna(subset=['idade - mulher', 'Tipo penal'])
    df_clean = df_clean[df_clean['Tipo penal'] != 'Não Informado']
    
    # Map crimes to categories
    df_clean['Crime_Category'] = df_clean['Tipo penal'].map(crime_to_category)
    df_clean['Crime_Category'] = df_clean['Crime_Category'].fillna('Múltiplas Dimensões (Outros)')
    
    # Create boxplot
    fig = px.box(
        df_clean,
        x='Crime_Category',
        y='idade - mulher',
        title='📊 Distribuição de Idade das Vítimas por Categoria de Crime',
        labels={'idade - mulher': 'Idade da Vítima', 'Crime_Category': 'Categoria de Crime'},
        color='Crime_Category',
        color_discrete_map={
            'Violência Física': '#e74c3c',
            'Violência Psicológica': '#f39c12',
            'Violência Moral': '#9b59b6',
            'Violência Sexual': '#e91e63',
            'Violência Econômica/Patrimonial': '#3498db',
            'Feminicídio': '#c0392b',
            'Múltiplas Dimensões (Outros)': '#95a5a6'
        }
    )
    
    fig.update_xaxes(tickangle=45)
    fig.update_layout(
        height=600,
        showlegend=False
    )
    
    # Add statistics annotation
    stats_text = []
    for category in df_clean['Crime_Category'].unique():
        cat_data = df_clean[df_clean['Crime_Category'] == category]['idade - mulher']
        if len(cat_data) > 0:
            stats_text.append(f"{category}: μ={cat_data.mean():.1f}, σ={cat_data.std():.1f}")
    
    fig.add_annotation(
        text="<br>".join(stats_text),
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        font=dict(size=10),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1
    )
    
    return fig

def create_violence_definition_box():
    """Create information box defining violence categories"""
    return """
    <div class="insight-box">
        <h4>📋 Definição: Categorias de Violência</h4>
        <p><strong>Violência Física</strong>: Lesão Corporal, Homicídio, Agressão, Tentativa de Homicídio</p>
        <p><strong>Violência Psicológica</strong>: Ameaça, Perseguição, Constrangimento Ilegal, Perturbação da Tranquilidade</p>
        <p><strong>Violência Moral</strong>: Injúria, Difamação, Calúnia</p>
        <p><strong>Violência Sexual</strong>: Estupro, Importunação Sexual, Assédio Sexual, Violação Sexual</p>
        <p><strong>Violência Econômica/Patrimonial</strong>: Dano, Furto, Roubo, Apropriação Indébita, Estelionato</p>
        <p><strong>Feminicídio</strong>: Feminicídio e Tentativa de Feminicídio</p>
        <p><strong>Múltiplas Dimensões</strong>: Outros crimes não categorizados acima</p>
    </div>
    """

@st.cache_data
def load_and_process_data(uploaded_file=None, file_path=None):
    """Enhanced data loading with comprehensive preprocessing"""
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ Arquivo carregado com sucesso! {len(df):,} registros encontrados.")
        except Exception as e:
            st.error(f"❌ Erro ao carregar arquivo: {e}")
            return None
    elif file_path and os.path.exists(file_path):
        try:
            df = pd.read_excel(file_path)
            st.success(f"✅ Arquivo carregado com sucesso! {len(df):,} registros encontrados.")
        except Exception as e:
            st.error(f"❌ Erro ao carregar arquivo: {e}")
            return None
    else:
        st.warning("⚠️ Nenhum arquivo carregado. Use a barra lateral para fazer upload.")
        return None
    
    # Enhanced data processing
    df = df.copy()
    
    # Process date column
    if 'Data do registro' in df.columns:
        df['Data do registro'] = pd.to_datetime(df['Data do registro'], errors='coerce')
        df['Ano'] = df['Data do registro'].dt.year
        df['Mês'] = df['Data do registro'].dt.month
        df['Dia_Semana'] = df['Data do registro'].dt.day_name()
        df['Trimestre'] = df['Data do registro'].dt.quarter
        df['Hora'] = df['Data do registro'].dt.hour
    
    # Clean and standardize text columns
    text_columns = ['Tipo penal', 'Vínculo', 'Escolaridade - mulher', 'Escolaridade - Agressor', 
                   'Município', 'Bairro', 'Nacionalidade - mulher', 'Nacionalidade - agressor']
    
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace({'nan': 'Não Informado', 'None': 'Não Informado'})
            df[col] = df[col].fillna('Não Informado')
            df[col] = df[col].str.strip()
            
            # Apply special standardization for education columns
            if 'Escolaridade' in col:
                df[col] = df[col].apply(standardize_education)
            else:
                df[col] = df[col].str.title()
    
    # Process age columns
    age_columns = ['idade - mulher', 'idade - agressor']
    for col in age_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].where((df[col] >= 10) & (df[col] <= 100))
    
    # Add derived columns
    if 'idade - mulher' in df.columns and 'idade - agressor' in df.columns:
        df['Diferença_Idade'] = df['idade - agressor'] - df['idade - mulher']
    
    # Create age groups
    if 'idade - mulher' in df.columns:
        df['Faixa_Etária_Vítima'] = pd.cut(df['idade - mulher'], 
                                          bins=[0, 18, 25, 35, 45, 55, 65, 100],
                                          labels=['Menor 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
    
    if 'idade - agressor' in df.columns:
        df['Faixa_Etária_Agressor'] = pd.cut(df['idade - agressor'], 
                                           bins=[0, 18, 25, 35, 45, 55, 65, 100],
                                           labels=['Menor 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'])
    
    # Add meso-region mapping
    if 'Município' in df.columns:
        df['Mesorregião'] = df['Município'].apply(get_mesoregion)
    
    # Create severity index based on crime type
    crime_severity = {
        'Feminicídio': 5, 'Estupro': 5, 'Lesão Corporal': 4, 'Ameaça': 3, 
        'Injúria': 2, 'Dano': 1, 'Violência Psicológica': 3
    }
    
    if 'Tipo penal' in df.columns:
        df['Índice_Gravidade'] = df['Tipo penal'].map(crime_severity).fillna(2)
    
    return df

def create_demographic_pyramid(df):
    """Create population pyramid style chart for age distribution"""
    if 'idade - mulher' not in df.columns or 'idade - agressor' not in df.columns:
        return None
    
    # Prepare data
    victim_ages = df['idade - mulher'].dropna()
    aggressor_ages = df['idade - agressor'].dropna()
    
    # Create age bins
    bins = range(10, 81, 5)
    victim_hist, _ = np.histogram(victim_ages, bins=bins)
    aggressor_hist, _ = np.histogram(aggressor_ages, bins=bins)
    
    # Create age labels
    age_labels = [f"{i}-{i+4}" for i in bins[:-1]]
    
    # Create pyramid data
    y_pos = np.arange(len(age_labels))
    
    fig = go.Figure()
    
    # Victims (left side, negative values)
    fig.add_trace(go.Bar(
        y=age_labels,
        x=-victim_hist,
        name='Vítimas',
        orientation='h',
        marker=dict(color='#FF6B6B'),
        hovertemplate='Faixa Etária: %{y}<br>Vítimas: %{x}<extra></extra>'
    ))
    
    # Aggressors (right side, positive values)
    fig.add_trace(go.Bar(
        y=age_labels,
        x=aggressor_hist,
        name='Agressores',
        orientation='h',
        marker=dict(color='#4ECDC4'),
        hovertemplate='Faixa Etária: %{y}<br>Agressores: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="🔍 Pirâmide Etária: Vítimas vs Agressores",
        title_font_size=16,
        xaxis_title="Número de Casos",
        yaxis_title="Faixa Etária",
        height=600,
        barmode='overlay',
        showlegend=True,
        template='plotly_white'
    )
    
    # Add center line
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    return fig

def create_temporal_analysis_enhanced(df):
    """Create enhanced temporal analysis focusing on day of week with viridis colors"""
    if 'Data do registro' not in df.columns:
        return None, None
    
    df_clean = df.dropna(subset=['Data do registro'])
    if len(df_clean) == 0:
        return None, None
    
    # Monthly trend
    df_clean = df_clean.copy()
    df_clean['Year'] = df_clean['Data do registro'].dt.year
    df_clean['Month'] = df_clean['Data do registro'].dt.month
    df_clean['DayOfWeek'] = df_clean['Data do registro'].dt.day_name()
    
    monthly_counts = df_clean.groupby(['Year', 'Month']).size().reset_index(name='count')
    monthly_counts['Date'] = pd.to_datetime(monthly_counts[['Year', 'Month']].assign(day=1))
    
    fig_monthly = px.line(
        monthly_counts,
        x='Date',
        y='count',
        title='📈 Tendência Mensal de Casos',
        markers=True,
        color_discrete_sequence=['#440154']  # Viridis purple
    )
    
    # Day of week analysis with proper order and viridis colors
    day_mapping = {
        'Monday': 'Segunda-feira',
        'Tuesday': 'Terça-feira', 
        'Wednesday': 'Quarta-feira',
        'Thursday': 'Quinta-feira',
        'Friday': 'Sexta-feira',
        'Saturday': 'Sábado',
        'Sunday': 'Domingo'
    }
    
    df_clean['DayOfWeek_PT'] = df_clean['DayOfWeek'].map(day_mapping)
    dow_counts = df_clean['DayOfWeek_PT'].value_counts()
    
    # Order days properly
    day_order_pt = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 
                    'Sexta-feira', 'Sábado', 'Domingo']
    dow_ordered = dow_counts.reindex([day for day in day_order_pt if day in dow_counts.index])
    
    fig_dow = px.bar(
        x=dow_ordered.index,
        y=dow_ordered.values,
        title='📊 Casos por Dia da Semana',
        labels={'x': 'Dia da Semana', 'y': 'Número de Casos'},
        color=dow_ordered.values,
        color_continuous_scale='Viridis'
    )
    
    fig_dow.update_layout(
        showlegend=False,
        xaxis_tickangle=45
    )
    
    return fig_monthly, fig_dow

def create_sankey_diagram(df):
    """Create Sankey diagram showing flow from relationship to crime type"""
    if 'Vínculo' not in df.columns or 'Tipo penal' not in df.columns:
        return None
    
    # Clean data
    df_clean = df.dropna(subset=['Vínculo', 'Tipo penal'])
    df_clean = df_clean[
        (df_clean['Vínculo'] != 'Não Informado') & 
        (df_clean['Tipo penal'] != 'Não Informado')
    ]
    
    if len(df_clean) == 0:
        return None
    
    # Get top categories (increased from 8 to 10 for better visibility)
    top_vinculos = df_clean['Vínculo'].value_counts().head(10).index.tolist()
    top_crimes = df_clean['Tipo penal'].value_counts().head(10).index.tolist()
    
    # Filter data
    df_filtered = df_clean[
        (df_clean['Vínculo'].isin(top_vinculos)) & 
        (df_clean['Tipo penal'].isin(top_crimes))
    ]
    
    # Create node labels
    all_labels = top_vinculos + top_crimes
    
    # Create source, target, and value lists
    source = []
    target = []
    value = []
    
    for vinculo in top_vinculos:
        for crime in top_crimes:
            count = len(df_filtered[
                (df_filtered['Vínculo'] == vinculo) & 
                (df_filtered['Tipo penal'] == crime)
            ])
            if count > 0:
                source.append(all_labels.index(vinculo))
                target.append(all_labels.index(crime))
                value.append(count)
    
    # Enhanced color scheme - more contrasting colors
    node_colors = (
        ['#FF6B6B', '#FF8E53', '#FF6B9D', '#C44569', '#F8B500', 
         '#6C5CE7', '#A29BFE', '#FD79A8', '#E17055', '#00B894'] +  # Vínculos
        ['#00CEC9', '#0984E3', '#6C5CE7', '#A29BFE', '#FD79A8', 
         '#E17055', '#00B894', '#FDCB6E', '#E84393', '#74B9FF']    # Crimes
    )
    
    # Ensure we have enough colors
    while len(node_colors) < len(all_labels):
        node_colors.extend(node_colors)
    
    node_colors = node_colors[:len(all_labels)]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=1.5),
            label=[f"<b>{label}</b>" for label in all_labels],  # Bold labels
            color=node_colors,
            hovertemplate='%{label}<br>Total: %{value}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=['rgba(0, 123, 191, 0.3)' for _ in value],  # Semi-transparent blue
            hovertemplate='%{source.label} → %{target.label}<br>Casos: %{value}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title="🔗 Fluxo: Tipo de Vínculo → Tipo de Crime",
        title_font_size=18,
        title_font_family="Arial Black",
        height=700,
        template='plotly_white',
        font=dict(size=14, family="Arial", color="black")
    )
    
    return fig

def create_education_comparison_enhanced(df):
    """Enhanced education level comparison with proper ordering"""
    if 'Escolaridade - mulher' not in df.columns or 'Escolaridade - Agressor' not in df.columns:
        return None
    
    # Define education order from lowest to highest
    education_order = [
        'Não Alfabetizado', 'Alfabetizado', 'Fundamental Incompleto',
        'Fundamental Completo', 'Médio Incompleto', 'Médio Completo',
        'Superior Incompleto', 'Superior Completo', 'Pós-graduação'
    ]
    
    # Clean data
    victim_edu = df['Escolaridade - mulher'].value_counts()
    aggressor_edu = df['Escolaridade - Agressor'].value_counts()
    
    # Get common education levels and order them
    all_education_levels = set(victim_edu.index) | set(aggressor_edu.index)
    all_education_levels = [edu for edu in all_education_levels if edu != 'Não Informado']
    
    # Order education levels
    ordered_education = []
    for level in education_order:
        if level in all_education_levels:
            ordered_education.append(level)
    
    # Add any remaining levels not in the predefined order
    for level in all_education_levels:
        if level not in ordered_education:
            ordered_education.append(level)
    
    # Create comparison dataframe
    education_df = pd.DataFrame({
        'Vítimas': [victim_edu.get(level, 0) for level in ordered_education],
        'Agressores': [aggressor_edu.get(level, 0) for level in ordered_education]
    }, index=ordered_education)
    
    # Calculate percentages
    education_df['Vítimas_Pct'] = education_df['Vítimas'] / education_df['Vítimas'].sum() * 100
    education_df['Agressores_Pct'] = education_df['Agressores'] / education_df['Agressores'].sum() * 100
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribuição Absoluta', 'Distribuição Percentual'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    # Absolute numbers
    fig.add_trace(
        go.Bar(x=education_df.index, y=education_df['Vítimas'], 
               name='Vítimas', marker_color='#FF6B6B'),
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(x=education_df.index, y=education_df['Agressores'], 
               name='Agressores', marker_color='#4ECDC4'),
        row=1, col=1
    )
    
    # Percentages
    fig.add_trace(
        go.Bar(x=education_df.index, y=education_df['Vítimas_Pct'], 
               name='Vítimas %', marker_color='#FF6B6B', showlegend=False),
        row=1, col=2
    )
    fig.add_trace(
        go.Bar(x=education_df.index, y=education_df['Agressores_Pct'], 
               name='Agressores %', marker_color='#4ECDC4', showlegend=False),
        row=1, col=2
    )
    
    fig.update_layout(
        title="📚 Análise de Escolaridade: Vítimas vs Agressores",
        height=600,
        barmode='group',
        template='plotly_white'
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_advanced_analytics_summary(df):
    """Create comprehensive analytics summary"""
    analytics = {}
    
    # Basic statistics
    analytics['total_cases'] = len(df)
    analytics['unique_municipalities'] = df['Município'].nunique() if 'Município' in df.columns else 0
    analytics['date_range'] = {
        'start': df['Data do registro'].min() if 'Data do registro' in df.columns else None,
        'end': df['Data do registro'].max() if 'Data do registro' in df.columns else None
    }
    
    # Age statistics
    if 'idade - mulher' in df.columns:
        analytics['victim_age_stats'] = {
            'mean': df['idade - mulher'].mean(),
            'median': df['idade - mulher'].median(),
            'std': df['idade - mulher'].std()
        }
    
    if 'idade - agressor' in df.columns:
        analytics['aggressor_age_stats'] = {
            'mean': df['idade - agressor'].mean(),
            'median': df['idade - agressor'].median(),
            'std': df['idade - agressor'].std()
        }
    
    # Crime patterns
    if 'Tipo penal' in df.columns:
        analytics['most_common_crime'] = df['Tipo penal'].mode()[0]
        analytics['crime_distribution'] = df['Tipo penal'].value_counts().to_dict()
    
    # Relationship patterns
    if 'Vínculo' in df.columns:
        analytics['most_common_relationship'] = df['Vínculo'].mode()[0]
        analytics['relationship_distribution'] = df['Vínculo'].value_counts().to_dict()
    
    # Temporal patterns
    if 'Data do registro' in df.columns:
        monthly_counts = df.groupby(df['Data do registro'].dt.month).size()
        analytics['peak_month'] = monthly_counts.idxmax()
        analytics['monthly_pattern'] = monthly_counts.to_dict()
    
    # Geographic concentration
    if 'Mesorregião' in df.columns:
        analytics['mesoregion_distribution'] = df['Mesorregião'].value_counts().to_dict()
        analytics['most_affected_region'] = df['Mesorregião'].mode()[0]
    
    return analytics

def create_folium_map(df, selected_crime='Todos os Crimes'):
    """Create interactive Folium map with correct mesoregion visualization"""
    if 'Mesorregião' not in df.columns:
        return None
    
    # Filter by crime type
    if selected_crime != 'Todos os Crimes':
        df_filtered = df[df['Tipo penal'] == selected_crime]
    else:
        df_filtered = df.copy()
    
    # Aggregate data by mesoregion
    meso_counts = df_filtered['Mesorregião'].value_counts().reset_index()
    meso_counts.columns = ['Mesorregião', 'Casos']
    
    # Create base map centered on Santa Catarina
    center_lat, center_lon = -27.2423, -50.2189
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='OpenStreetMap')
    
    # Define mesoregion approximate centers
    mesoregion_centers = {
        'Oeste Catarinense': [-26.9, -52.8],
        'Norte Catarinense': [-26.3, -49.0],
        'Vale do Itajaí': [-26.9, -49.0],
        'Serrana': [-27.8, -50.3],
        'Grande Florianópolis': [-27.5, -48.5],
        'Sul Catarinense': [-28.5, -49.3]
    }
    
    # Add markers for each mesoregion
    max_cases = meso_counts['Casos'].max()
    
    for _, row in meso_counts.iterrows():
        if row['Mesorregião'] in mesoregion_centers:
            lat, lon = mesoregion_centers[row['Mesorregião']]
            
            # Scale marker size based on cases
            radius = 20 + (row['Casos'] / max_cases * 30)
            
            # Color intensity based on number of cases
            color_intensity = row['Casos'] / max_cases
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=f"<b>{row['Mesorregião']}</b><br>Casos: {row['Casos']}",
                tooltip=f"{row['Mesorregião']}: {row['Casos']} casos",
                color='darkred',
                fill=True,
                fillColor='red',
                fillOpacity=0.3 + (color_intensity * 0.6)
            ).add_to(m)
            
            # Add label
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(html=f"""
                    <div style="font-size: 12pt; color: black; font-weight: bold; 
                               text-shadow: 1px 1px 1px white;">
                        {row['Mesorregião']}<br>{row['Casos']} casos
                    </div>""")
            ).add_to(m)
    
    # Add a legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: 120px; 
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="margin: 0;"><b>Legenda - Mesorregiões</b></p>
    <p style="margin: 0;">🔴 Tamanho = Nº casos</p>
    <p style="margin: 0;">Cor = Intensidade</p>
    <p style="margin: 0; font-size: 12px;"><i>{}</i></p>
    </div>
    '''.format(selected_crime)
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def export_to_excel(data, filename, sheet_name='Dados'):
    """Export data to Excel format"""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if isinstance(data, dict):
            for sheet, df in data.items():
                if isinstance(df, pd.DataFrame):
                    df.to_excel(writer, sheet_name=sheet, index=True)
        elif isinstance(data, pd.DataFrame):
            data.to_excel(writer, sheet_name=sheet_name, index=True)
        else:
            # Convert to DataFrame if possible
            try:
                pd.DataFrame(data).to_excel(writer, sheet_name=sheet_name, index=True)
            except:
                st.error("Não foi possível converter os dados para Excel")
                return None
    
    output.seek(0)
    return output

def main():
    st.markdown('<h1 class="main-header">🔍 Análise de Violência contra a Mulher - Santa Catarina</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header('📁 Configurações')
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "📊 Carregar arquivo Excel",
        type=['xlsx', 'xls'],
        help="Faça upload do arquivo scmulherextracao2025.xlsx"
    )
    
    # Use local file option
    use_local_file = st.sidebar.checkbox("🔗 Usar arquivo local")
    local_file_path = None
    
    if use_local_file:
        local_file_path = st.sidebar.text_input(
            "Caminho do arquivo:",
            value="scmulher-extracao-2025.xlsx"
        )
    
    # Load data
    with st.spinner('🔄 Carregando e processando dados...'):
        df = load_and_process_data(uploaded_file, local_file_path if use_local_file else None)
    
    if df is None:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ Dados não carregados</h3>
            <p>Por favor, carregue um arquivo Excel para começar a análise.</p>
            <p>Use a barra lateral para fazer upload do arquivo ou especificar o caminho local.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Generate analytics summary
    analytics = create_advanced_analytics_summary(df)
    
    # Overview metrics
    st.markdown('<div class="success-box"><h3>📊 Visão Geral do Dataset</h3></div>', 
                unsafe_allow_html=True)
    
    # Show analysis period if available
    if 'date_range' in analytics and analytics['date_range']['start'] and analytics['date_range']['end']:
        st.info(f"📅 **Período de Análise**: {analytics['date_range']['start'].strftime('%d/%m/%Y')} até {analytics['date_range']['end'].strftime('%d/%m/%Y')}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "📋 Total de Casos", 
            f"{analytics['total_cases']:,}",
            help="Número total de registros no dataset"
        )
    
    with col2:
        st.metric(
            "🏘️ Municípios", 
            analytics['unique_municipalities'],
            help="Número de municípios com registros"
        )
    
    with col3:
        if 'victim_age_stats' in analytics:
            st.metric(
                "👩 Idade Média Vítimas", 
                f"{analytics['victim_age_stats']['mean']:.1f} anos",
                help="Idade média das vítimas"
            )
        else:
            st.metric("👩 Idade Média Vítimas", "N/A")
    
    with col4:
        if 'most_common_crime' in analytics:
            st.metric(
                "⚖️ Crime Mais Comum", 
                analytics['most_common_crime'],
                help="Tipo de crime mais frequente"
            )
        else:
            st.metric("⚖️ Crime Mais Comum", "N/A")
    
    with col5:
        if 'most_affected_region' in analytics:
            st.metric(
                "📍 Região Mais Afetada", 
                analytics['most_affected_region'],
                help="Mesorregião com mais casos"
            )
        else:
            st.metric("📍 Região Mais Afetada", "N/A")
    
    # Filters
    st.sidebar.header('🔍 Filtros Avançados')
    
    # Date filter
    if 'Data do registro' in df.columns and df['Data do registro'].notna().any():
        min_date = df['Data do registro'].min().date()
        max_date = df['Data do registro'].max().date()
        date_range = st.sidebar.date_input(
            '📅 Período de Análise',
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = None
    
    # Crime type filter
    if 'Tipo penal' in df.columns:
        available_crimes = df['Tipo penal'].value_counts().index.tolist()
        crime_types = st.sidebar.multiselect(
            '⚖️ Tipos de Crime',
            options=available_crimes,
            default=available_crimes[:10] if len(available_crimes) > 10 else available_crimes
        )
    else:
        crime_types = []
    
    # Meso-region filter
    if 'Mesorregião' in df.columns:
        available_regions = df['Mesorregião'].value_counts().index.tolist()
        selected_regions = st.sidebar.multiselect(
            '🗺️ Mesorregiões',
            options=available_regions,
            default=available_regions
        )
    else:
        selected_regions = []
    
    # Apply filters
    filtered_df = df.copy()
    
    if date_range and 'Data do registro' in df.columns:
        if len(date_range) == 2:
            mask = (
                (filtered_df['Data do registro'].dt.date >= date_range[0]) & 
                (filtered_df['Data do registro'].dt.date <= date_range[1])
            )
            filtered_df = filtered_df[mask]
    
    if crime_types and 'Tipo penal' in df.columns:
        filtered_df = filtered_df[filtered_df['Tipo penal'].isin(crime_types)]
    
    if selected_regions and 'Mesorregião' in df.columns:
        filtered_df = filtered_df[filtered_df['Mesorregião'].isin(selected_regions)]
    
    # Show filter results
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Registros Filtrados", f"{len(filtered_df):,}")
    
    if len(filtered_df) == 0:
        st.warning("⚠️ Nenhum registro encontrado com os filtros selecionados.")
        st.stop()
    
    # Main analysis tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        '🏠 Home', 
        '👥 Demografia', 
        '📅 Padrões Temporais', 
        '🔗 Relacionamentos',
        '🗺️ Análise Geográfica',
        '📋 Análise Questionário'
    ])
    
    with tab1:
        st.header('🎯 Análise Geral')
        
        # Summary table with proper multi-index
        st.subheader('📋 Tabela Resumo Multi-Index')
        
        try:
            summary_data, multi_index_summary = create_multi_index_summary_table(filtered_df)
            if summary_data is not None and len(summary_data) > 0:
                
                # Display the multi-index table if available
                if multi_index_summary is not None:
                    st.subheader('📊 Tabela de Categorias por Tipo de Crime')
                    
                    # Convert to HTML for better styling
                    html_table = multi_index_summary.to_html(
                        classes='dataframe',
                        table_id='summary_table',
                        escape=False
                    )
                    
                    # Add custom CSS for the table
                    st.markdown("""
                    <style>
                    #summary_table {
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }
                    #summary_table th {
                        background-color: #f0f2f6;
                        font-weight: bold;
                        padding: 10px;
                        text-align: center;
                        border: 1px solid #ddd;
                    }
                    #summary_table td {
                        padding: 8px;
                        text-align: center;
                        border: 1px solid #ddd;
                    }
                    #summary_table tbody tr th {
                        background-color: #e0e3e9;
                        font-weight: bold;
                        text-align: left;
                        padding-left: 10px;
                    }
                    #summary_table tbody tr th.level0 {
                        background-color: #d4d7dd;
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    # Display the table
                    st.markdown(html_table, unsafe_allow_html=True)
                    
                    # Alternative: Use st.dataframe with styling
                    st.subheader('📊 Visualização Alternativa (Interativa)')
                    st.dataframe(multi_index_summary, use_container_width=True, height=600)
                else:
                    # Fallback to regular table
                    st.subheader('📊 Tabela Resumo Detalhada')
                    st.dataframe(summary_data.head(50), use_container_width=True)
                
                # Export summary table
                if st.button("📥 Exportar Tabela Resumo", key="export_summary"):
                    export_data = {}
                    if multi_index_summary is not None:
                        export_data['Tabela_Multi_Index'] = multi_index_summary
                    export_data['Resumo_Detalhado'] = summary_data
                    
                    excel_data = export_to_excel(export_data, 'tabela_resumo.xlsx')
                    if excel_data:
                        st.download_button(
                            label="⬇️ Download Excel - Tabela Resumo",
                            data=excel_data,
                            file_name=f"tabela_resumo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.info("Dados insuficientes para gerar tabela resumo")
        except Exception as e:
            st.error(f"Erro ao criar tabela resumo: {e}")
            st.info("Continuando sem a tabela resumo...")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Crime distribution
            if 'Tipo penal' in filtered_df.columns:
                crime_counts = filtered_df['Tipo penal'].value_counts()
                fig = px.pie(
                    values=crime_counts.values,
                    names=crime_counts.index,
                    title='📊 Distribuição por Tipo de Crime',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Relationship distribution
            if 'Vínculo' in filtered_df.columns:
                relationship_counts = filtered_df['Vínculo'].value_counts()
                fig = px.pie(
                    values=relationship_counts.values,
                    names=relationship_counts.index,
                    title='💕 Distribuição por Tipo de Vínculo',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)
        
        # Sankey diagram
        st.subheader('🔗 Fluxo: Vínculo → Crime')
        sankey_fig = create_sankey_diagram(filtered_df)
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True)
        else:
            st.info("Dados insuficientes para o diagrama Sankey")
    
    with tab2:
        st.header('👥 Análise Demográfica')
        
        # Violence definition box
        st.markdown(create_violence_definition_box(), unsafe_allow_html=True)
        
        # Demographic pyramid
        pyramid_fig = create_demographic_pyramid(filtered_df)
        if pyramid_fig:
            st.plotly_chart(pyramid_fig, use_container_width=True)
        else:
            st.info("Dados de idade insuficientes para a pirâmide demográfica")
        
        # Race/Ethnicity analysis
        st.subheader('🌍 Análise por Raça/Cor')
        race_fig, race_data = create_race_analysis(filtered_df)
        if race_fig:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.plotly_chart(race_fig, use_container_width=True)
            with col2:
                st.dataframe(race_data, use_container_width=True)
        else:
            st.info("Dados de raça/cor não disponíveis (q26)")
        
        # Nationality analysis (if not all Brazilian)
        if 'Nacionalidade - mulher' in filtered_df.columns:
            nationality_counts = filtered_df['Nacionalidade - mulher'].value_counts()
            if len(nationality_counts) > 1 or (len(nationality_counts) == 1 and nationality_counts.index[0] != 'Brasileira'):
                st.subheader('🌐 Análise por Nacionalidade')
                col1, col2 = st.columns([2, 1])
                with col1:
                    fig_nat = px.bar(
                        x=nationality_counts.index,
                        y=nationality_counts.values,
                        title='Nacionalidade das Vítimas',
                        labels={'x': 'Nacionalidade', 'y': 'Quantidade'},
                        color_discrete_sequence=['#3498db']
                    )
                    st.plotly_chart(fig_nat, use_container_width=True)
                with col2:
                    nat_df = pd.DataFrame({
                        'Nacionalidade': nationality_counts.index,
                        'Quantidade': nationality_counts.values,
                        '%': (nationality_counts.values / len(filtered_df) * 100).round(1)
                    })
                    st.dataframe(nat_df, use_container_width=True)
        
        # Education comparison
        edu_fig = create_education_comparison_enhanced(filtered_df)
        if edu_fig:
            st.plotly_chart(edu_fig, use_container_width=True)
        else:
            st.info("Dados de escolaridade insuficientes")
        
        # Crime category boxplot
        st.subheader('📊 Idade das Vítimas por Categoria de Crime')
        category_boxplot = create_crime_category_boxplot(filtered_df)
        if category_boxplot:
            st.plotly_chart(category_boxplot, use_container_width=True)
        else:
            st.info("Dados insuficientes para análise por categoria de crime")
    
    with tab3:
        st.header('📅 Padrões Temporais')
        
        # Temporal analysis with enhanced day of week chart
        monthly_fig, dow_fig = create_temporal_analysis_enhanced(filtered_df)
        
        if monthly_fig:
            st.plotly_chart(monthly_fig, use_container_width=True)
        else:
            st.info("Dados temporais insuficientes")
        
        if dow_fig:
            st.plotly_chart(dow_fig, use_container_width=True)
        else:
            st.info("Dados temporais insuficientes para análise por dia da semana")
        
        # Seasonality analysis
        st.subheader('🌡️ Análise de Sazonalidade')
        season_fig1, season_fig2 = create_seasonality_analysis(filtered_df)
        
        if season_fig1:
            st.plotly_chart(season_fig1, use_container_width=True)
            
            # Insights about seasonality
            st.markdown("""
            <div class="insight-box">
                <h4>📊 Insights de Sazonalidade</h4>
                <p>• Meses em <b style="color: red;">vermelho</b> apresentam casos acima da média + 1 desvio padrão</p>
                <p>• A linha <b style="color: green;">verde</b> indica a média mensal</p>
                <p>• A linha <b style="color: red;">vermelha pontilhada</b> indica o limite superior (média + desvio padrão)</p>
            </div>
            """, unsafe_allow_html=True)
            
            if season_fig2:
                st.plotly_chart(season_fig2, use_container_width=True)
        else:
            st.info("Dados insuficientes para análise de sazonalidade")
    
    with tab4:
        st.header('🔗 Análise de Relacionamentos')
        
        # Relationship vs Crime matrix
        if 'Vínculo' in filtered_df.columns and 'Tipo penal' in filtered_df.columns:
            clean_df = filtered_df.dropna(subset=['Vínculo', 'Tipo penal'])
            
            if len(clean_df) > 0:
                matrix = pd.crosstab(clean_df['Vínculo'], clean_df['Tipo penal'])
                
                fig = px.imshow(
                    matrix.values,
                    x=matrix.columns,
                    y=matrix.index,
                    title='🔥 Mapa de Calor: Tipo de Vínculo vs Tipo de Crime',
                    color_continuous_scale='Reds',
                    aspect='auto'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
        
        # Education Correlation Heatmap
        st.subheader('🎓 Mapa de Calor: Escolaridade Vítima vs Agressor')
        
        edu_heatmap = create_education_correlation_heatmap(filtered_df)
        if edu_heatmap:
            st.plotly_chart(edu_heatmap, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
                <h4>📊 Como interpretar este mapa de calor</h4>
                <p>• Cores mais intensas indicam maior número de casos</p>
                <p>• Eixo X: Escolaridade do Agressor</p>
                <p>• Eixo Y: Escolaridade da Vítima</p>
                <p>• Diagonal: Casos onde vítima e agressor têm mesma escolaridade</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Dados insuficientes para criar mapa de calor de escolaridade")
        
        # Education-Crime Correlation Analysis
        st.subheader('📊 Análise de Correlação: Escolaridade vs Ocorrência de Crimes')
        
        corr_fig, corr_data = create_education_crime_correlation(filtered_df)
        if corr_fig:
            st.plotly_chart(corr_fig, use_container_width=True)
            
            # Interpretation
            if corr_data and 'correlation' in corr_data:
                correlation = corr_data['correlation']
                p_value = corr_data['p_value']
                
                if p_value < 0.05:
                    if correlation < 0:
                        st.success(f"✅ **Correlação Negativa Significativa**: r = {correlation:.3f}, p < 0.05. "
                                 "Menor escolaridade está associada a maior número de crimes.")
                    else:
                        st.warning(f"⚠️ **Correlação Positiva**: r = {correlation:.3f}, p < 0.05. "
                                 "Maior escolaridade está associada a maior número de crimes.")
                else:
                    st.info(f"ℹ️ **Sem Correlação Significativa**: r = {correlation:.3f}, p = {p_value:.3f}")
        
        # Age difference analysis
        if 'Diferença_Idade' in filtered_df.columns:
            age_diff_data = filtered_df['Diferença_Idade'].dropna()
            
            if len(age_diff_data) > 0:
                fig = px.histogram(
                    age_diff_data,
                    title='📊 Distribuição da Diferença de Idade (Agressor - Vítima)',
                    labels={'value': 'Diferença de Idade (anos)', 'count': 'Frequência'},
                    nbins=50,
                    color_discrete_sequence=['#FF6B6B']
                )
                fig.add_vline(x=age_diff_data.mean(), line_dash="dash", 
                             annotation_text=f"Média: {age_diff_data.mean():.1f} anos")
                st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header('🗺️ Análise Geográfica')
        
        # Crime selection with session state
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'Tipo penal' in filtered_df.columns:
                available_crimes = ['Todos os Crimes'] + filtered_df['Tipo penal'].value_counts().index.tolist()
                selected_crime = st.selectbox(
                    '⚖️ Selecione o Tipo de Crime para o Mapa:',
                    options=available_crimes,
                    key='crime_selector',
                    index=available_crimes.index(st.session_state.selected_crime) if st.session_state.selected_crime in available_crimes else 0,
                    help="Escolha um tipo específico de crime ou 'Todos os Crimes' para visualizar o total"
                )
                st.session_state.selected_crime = selected_crime
        
        # Municipality heatmap
        st.subheader('🏙️ Mapa de Calor por Municípios')
        
        municipality_map = create_municipality_heatmap(filtered_df, selected_crime)
        if municipality_map:
            folium_html = municipality_map._repr_html_()
            components.html(folium_html, height=600)
        else:
            st.info("Mapa de calor por municípios não disponível")
        
        # Mesoregion choropleth
        st.subheader('🗺️ Distribuição por Mesorregiões')
        choropleth_fig, meso_data = create_mesoregion_choropleth(filtered_df, selected_crime)
        
        if choropleth_fig:
            st.plotly_chart(choropleth_fig, use_container_width=True)
            
            # Statistics by region
            if meso_data is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🗺️ Mesorregiões Afetadas", len(meso_data))
                with col2:
                    st.metric("📊 Total de Casos", meso_data['Casos'].sum())
                with col3:
                    top_region = meso_data.iloc[0] if len(meso_data) > 0 else None
                    if top_region is not None:
                        st.metric("🥇 Região com Mais Casos", f"{top_region['Mesorregião']}")
                
                # Show data table
                meso_data['Porcentagem'] = (meso_data['Casos'] / meso_data['Casos'].sum() * 100).round(2)
                meso_data['Ranking'] = range(1, len(meso_data) + 1)
                display_meso = meso_data[['Ranking', 'Mesorregião', 'Casos', 'Porcentagem']]
                st.dataframe(display_meso, use_container_width=True)
        
        # Interactive map with hover
        st.subheader('🌐 Mapa Interativo Mesorregiões com Hover')
        folium_map = create_folium_map(filtered_df, selected_crime)
        if folium_map:
            folium_html = folium_map._repr_html_()
            components.html(folium_html, height=500)
        else:
            st.info("Mapa interativo não disponível")
        
        # Top municipalities table
        if 'Município' in filtered_df.columns:
            st.subheader(f'🏆 Top 15 Municípios - {selected_crime}')
            
            if selected_crime != 'Todos os Crimes':
                df_muni = filtered_df[filtered_df['Tipo penal'] == selected_crime]
            else:
                df_muni = filtered_df.copy()
            
            muni_counts = df_muni['Município'].value_counts().head(15)
            muni_df = pd.DataFrame({
                'Ranking': range(1, len(muni_counts) + 1),
                'Município': muni_counts.index,
                'Casos': muni_counts.values,
                'Porcentagem': (muni_counts.values / len(df_muni) * 100).round(2)
            })
            st.dataframe(muni_df, use_container_width=True)
    
    with tab6:
        st.header('📋 Análise do Questionário FNAV')
        
        # Response frequency table
        st.subheader('📊 Frequência de Respostas por Pergunta')
        
        response_table = create_questionnaire_response_table(filtered_df)
        if response_table is not None and len(response_table) > 0:
            st.markdown("""
            <div class="insight-box">
                <h4>📋 Como interpretar esta tabela</h4>
                <p>• <b>Pergunta</b>: Número da questão (q1, q2, etc.)</p>
                <p>• <b>Alternativa</b>: Opção selecionada (0, 1, 2, etc.)</p>
                <p>• <b>N</b>: Número absoluto de respostas</p>
                <p>• <b>%</b>: Percentual em relação ao total de respondentes</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display table with proper formatting
            st.dataframe(
                response_table.style.format({
                    'N': '{:,}',
                    '%': '{:.1f}%'
                }),
                use_container_width=True,
                height=400
            )
            
            # Export option
            if st.button("📥 Exportar Tabela de Respostas", key="export_responses"):
                excel_data = export_to_excel(response_table, 'respostas_questionario.xlsx', 'Respostas')
                if excel_data:
                    st.download_button(
                        label="⬇️ Download Excel - Respostas",
                        data=excel_data,
                        file_name=f"respostas_questionario_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        else:
            st.info("Dados de questionário não disponíveis")
        
        # Psychological indicators analysis
        st.subheader('🧠 Indicadores Psicológicos')
        
        psych_results = analyze_psychological_indicators(filtered_df)
        
        # Q34 Analysis
        if 'q34' in psych_results:
            st.markdown("""
            <div class="warning-box">
                <h4>Q34: Apresentação Física e Emocional</h4>
                <p>Esta questão avalia sinais de esgotamento emocional, uso de medicação controlada 
                e necessidade de acompanhamento psicológico/psiquiátrico.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(psych_results['q34']['fig'], use_container_width=True)
        
        # Q35 Analysis
        if 'q35' in psych_results:
            st.markdown("""
            <div class="warning-box">
                <h4>Q35: Risco de Suicídio</h4>
                <p>Esta questão aborda ideação ou tentativa suicida, critério essencial de risco extremo.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(psych_results['q35']['fig'], use_container_width=True)
            
            # Show statistics
            if 'data' in psych_results['q35']:
                data = psych_results['q35']['data']
                if len(data) > 0:
                    risk_cases = data.get('Sim', 0) if 'Sim' in data.index else 0
                    total_cases = data.sum()
                    risk_percentage = (risk_cases / total_cases * 100) if total_cases > 0 else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⚠️ Casos com Risco", f"{risk_cases:,}")
                    with col2:
                        st.metric("📊 Total Avaliado", f"{total_cases:,}")
                    with col3:
                        st.metric("📈 Percentual de Risco", f"{risk_percentage:.1f}%")
        
        # Risk Score Analysis
        st.subheader('⚡ Análise de Escore de Risco')
        
        risk_fig, risk_data = create_risk_score_analysis(filtered_df.copy())
        if risk_fig:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(risk_fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="insight-box">
                    <h4>🎯 Fatores de Risco Considerados</h4>
                    <ul>
                        <li>Ameaça com arma de fogo (peso: 3)</li>
                        <li>Ameaça com faca (peso: 2)</li>
                        <li>Tiro (peso: 3)</li>
                        <li>Facada (peso: 3)</li>
                        <li>Risco de suicídio (peso: 5)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk level distribution
                if 'risk_level' in risk_data.columns:
                    risk_dist = risk_data['risk_level'].value_counts()
                    risk_df = pd.DataFrame({
                        'Nível': risk_dist.index,
                        'Casos': risk_dist.values,
                        '%': (risk_dist.values / len(risk_data) * 100).round(1)
                    })
                    st.dataframe(risk_df, use_container_width=True)
        else:
            st.info("Dados insuficientes para calcular escore de risco")
        
        # Correlation with sociodemographic data
        st.subheader('🔗 Correlação com Dados Sociodemográficos')
        
        st.markdown("""
        <div class="insight-box">
            <h4>📊 Análises Disponíveis</h4>
            <p>• Cruzamento de indicadores de risco com idade das vítimas</p>
            <p>• Relação entre escolaridade e presença de fatores de risco</p>
            <p>• Distribuição geográfica dos casos de alto risco</p>
            <p>• Padrões de reincidência e evolução do risco</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional analyses can be added here based on specific requirements
    
    # Download options
    st.sidebar.header('📥 Downloads')
    
    # Download filtered data
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Baixar dados filtrados (CSV)",
        data=csv,
        file_name=f'violencia_sc_filtrado_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv'
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📊 <strong>Análise de Violência contra a Mulher - Santa Catarina</strong></p>
        <p>Desenvolvido com ❤️ usando Streamlit, Plotly e outras tecnologias modernas</p>
        <p>🔒 Dados tratados com responsabilidade e confidencialidade</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()