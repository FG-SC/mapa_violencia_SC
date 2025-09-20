import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import geobr
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components
import warnings
from pandas.api.types import CategoricalDtype
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard - Violência contra a Mulher SC",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
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
    .dataframe th {
        background-color: #f0f2f6 !important;
        font-weight: bold !important;
    }
    .crime-category-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
        border-left: 4px solid #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """
    Carrega e processa os dados do arquivo CSV com cache para otimização.
    """
    try:
        # Carrega o arquivo CSV
        df = pd.read_csv('dados_processados_final.csv', index_col=0)
        
        # Aplique a função à coluna para a conversão
        df['data_nascimento_agressor'] = pd.to_datetime(df['data_nascimento_agressor'], format='%d-%m-%Y')
        # Aplique a função à coluna para a conversão
        df['data_nascimento_mulher'] = pd.to_datetime(df['data_nascimento_mulher'], format='%d-%m-%Y')

        # Converte a coluna de data para datetime
        df['data_denuncia'] = pd.to_datetime(df['data_denuncia'])
                
        # Limpa e padroniza dados
        string_columns = ['Nome município IBGE', 'bairro', 'vinculo', 'sexo',
                           'nome_escolaridade_mulher', 'sexo_agressor', 
                         'nome_escolaridade_agressor', 'titulo_fato', 
                         'Mesorregião IBGE']
        
        df.rename(columns={'Nome município IBGE': 'municipio_corrigido'}, inplace=True)

        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', ''], 'Não Informado')
        
        # Calcula idade das vítimas e agressores se as colunas de nascimento existirem
        if 'data_nascimento_mulher' in df.columns:
            df['idade_mulher'] = (df['data_denuncia'] - df['data_nascimento_mulher']).dt.days // 365
            df['idade_mulher'] = df['idade_mulher'].where((df['idade_mulher'] >= 0) & (df['idade_mulher'] <= 120))
        
        if 'data_nascimento_agressor' in df.columns:
            df['idade_agressor'] = (df['data_denuncia'] - df['data_nascimento_agressor']).dt.days // 365
            df['idade_agressor'] = df['idade_agressor'].where((df['idade_agressor'] >= 0) & (df['idade_agressor'] <= 120))
        
        # Definir ordem lógica dos níveis de escolaridade
        education_order = [
            'Não Alfabetizado', 'Semialfabetizado',
            'Ensino fundamental incompleto', 'Ensino fundamental completo',
            'Ensino médio incompleto', 'Ensino Médio Completo',
            'Superior incompleto', 'Superior (cursando)', 'Superior completo',
            'Pós Graduação', 'Mestrado', 'Doutorado'
        ]
        
        # Criar tipo categórico com ordem definida
        cat_type = CategoricalDtype(categories=education_order, ordered=True)
        
        # Aplicar ordenação categórica às colunas de escolaridade
        if 'nome_escolaridade_mulher' in df.columns:
            df['nome_escolaridade_mulher'] = df['nome_escolaridade_mulher'].astype(cat_type)
        
        if 'nome_escolaridade_agressor' in df.columns:
            df['nome_escolaridade_agressor'] = df['nome_escolaridade_agressor'].astype(cat_type)
        
        # Criar coluna de cor/raça da vítima a partir de colunas one-hot
        colunas_raca = ['Branca', 'Preta', 'Parda', 'Amarela', 'Indígena']
        # Verificar se as colunas de raça existem no dataframe
        if all(col in df.columns for col in colunas_raca):
            def definir_cor_raca(row):
                for col in colunas_raca:
                    if pd.notna(row[col]) and row[col] == 1.0:
                        return col
                return 'Não Informado'
            df['cor_raca_vitima'] = df.apply(definir_cor_raca, axis=1)
        
        return df
        
    except FileNotFoundError:
        st.error("⚠ Arquivo 'dados_processados.csv' não encontrado. Verifique se o arquivo está no diretório correto.")
        return None
    except Exception as e:
        st.error(f"⚠ Erro ao carregar os dados: {str(e)}")
        return None

def categorize_crime(df):
    """
    Adiciona categorização de crimes ao DataFrame.
    """
    crime_mapping = {
    'Violência Física': ['Lesão corporal'],
    'Violência Psicológica': ['Ameaça', 'Constrangimento ilegal', 'Violência Psicológica'],
    'Violência Moral': ['Calúnia', 'Difamação', 'Injúria'],
    'Violência Sexual': ['Assédio sexual', 'Estupro', 'Sedução'],
    'Violência Patrimonial': ['Apropriação indébita', 'Dano', 'Furto', 'Roubo'],
    'Violência de Múltipla Dimensões': ['Outro', 'Sequestro e Carcere Privado']
}
    
    df_copy = df.copy()
    df_copy['categoria_crime'] = 'Outro'
    
    for categoria, crimes in crime_mapping.items():
        mask = df_copy['titulo_fato'].isin(crimes)
        df_copy.loc[mask, 'categoria_crime'] = categoria
    
    return df_copy

def calculate_kpis(df):
    """
    Calcula os KPIs principais do dashboard.
    """
    total_denuncias = len(df)
    
    # Vítimas únicas
    if 'id_mulher' in df.columns:
        vitimas_unicas = df['id_mulher'].nunique()
    else:
        vitimas_unicas = "N/A"
    
    # Agressores únicos
    if 'id_agressor' in df.columns:
        agressores_unicos = df['id_agressor'].nunique()
    else:
        agressores_unicos = "N/A"
    
    return total_denuncias, vitimas_unicas, agressores_unicos

def create_mesoregion_choropleth_map(df):
    """
    Cria mapa coroplético por mesorregião de SC.
    """
    if 'Mesorregião IBGE' not in df.columns:
        return None
    
    try:
        # Carregar dados geográficos das mesorregiões de SC
        sc_meso = geobr.read_meso_region(code_meso="SC", year=2020)
        
        # Contar casos por mesorregião
        meso_counts = df['Mesorregião IBGE'].value_counts().reset_index()
        meso_counts.columns = ['Mesorregião', 'casos']
        
        # Merge dos dados
        merged_data = sc_meso.merge(
            meso_counts, 
            left_on='name_meso', 
            right_on='Mesorregião', 
            how='left'
        )
        merged_data['casos'] = merged_data['casos'].fillna(0)
        
        # Criar mapa base
        center_lat, center_lon = -27.2423, -50.2189
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles='OpenStreetMap')
        
        # Adicionar choropleth
        folium.Choropleth(
            geo_data=merged_data.to_json(),
            name='choropleth',
            data=merged_data,
            columns=['code_meso', 'casos'],
            key_on='feature.properties.code_meso',
            fill_color='BuPu',
            fill_opacity=0.8,
            line_opacity=0.3,
            legend_name='Número de Casos por Mesorregião'
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        return m
        
    except Exception as e:
        st.warning(f"Erro ao criar mapa de mesorregiões: {str(e)}")
        return None

def create_race_distribution_chart(df):
    """
    Cria gráfico de distribuição por raça/cor da vítima.
    """
    if 'cor_raca_vitima' not in df.columns:
        return None
    
    race_counts = df['cor_raca_vitima'].value_counts()
    
    fig = px.bar(
        race_counts, 
        y=race_counts.values, 
        x=race_counts.index,
        title='📊 Distribuição de Vítimas por Raça/Cor',
        labels={'y': 'Número de Casos', 'x': 'Raça/Cor'},
        color=race_counts.index,
        text_auto=True
    )
    fig.update_layout(showlegend=False, height=500)
    return fig

def create_race_crime_heatmap(df):
    """
    Cria heatmap cruzando raça/cor da vítima com categoria de crime.
    """
    if 'cor_raca_vitima' not in df.columns or 'categoria_crime' not in df.columns:
        return None
    
    # Filtrar dados válidos
    df_clean = df.dropna(subset=['cor_raca_vitima', 'categoria_crime'])
    df_clean = df_clean[df_clean['cor_raca_vitima'] != 'Não Informado']
    
    if len(df_clean) == 0:
        return None
    
    crosstab = pd.crosstab(df_clean['cor_raca_vitima'], df_clean['categoria_crime'])
    
    fig = px.imshow(
        crosstab, 
        text_auto=True, 
        aspect="auto",
        title='🔥 Mapa de Calor: Raça/Cor da Vítima vs. Tipo de Crime',
        labels=dict(x="Tipo de Crime", y="Raça/Cor da Vítima", color="Nº de Casos"),
        color_continuous_scale='Reds'
    )
    fig.update_layout(height=600, xaxis_tickangle=45)
    return fig

def create_sankey_diagram(df):
    """
    Cria diagrama de Sankey mostrando o fluxo de Vínculo -> Tipo de Crime.
    """
    if 'vinculo' not in df.columns or 'titulo_fato' not in df.columns:
        return None
    
    # Filtrar dados válidos
    df_clean = df.dropna(subset=['vinculo', 'titulo_fato'])
    df_clean = df_clean[
        (df_clean['vinculo'] != 'Não Informado') & 
        (df_clean['titulo_fato'] != 'Não Informado')
    ]
    
    if len(df_clean) == 0:
        return None
    
    # Obter top 10 vínculos e top 10 crimes (aumentado para melhor visibilidade)
    top_vinculos = df_clean['vinculo'].value_counts().head(10).index.tolist()
    top_crimes = df_clean['titulo_fato'].value_counts().head(10).index.tolist()
    
    # Filtrar dataset para incluir apenas os tops
    df_filtered = df_clean[
        (df_clean['vinculo'].isin(top_vinculos)) & 
        (df_clean['titulo_fato'].isin(top_crimes))
    ]
    
    if len(df_filtered) == 0:
        return None
    
    # Criar lista de todos os labels
    all_labels = top_vinculos + top_crimes
    
    # Criar listas source, target e value
    source = []
    target = []
    value = []
    
    for vinculo in top_vinculos:
        for crime in top_crimes:
            count = len(df_filtered[
                (df_filtered['vinculo'] == vinculo) & 
                (df_filtered['titulo_fato'] == crime)
            ])
            if count > 0:
                source.append(all_labels.index(vinculo))
                target.append(all_labels.index(crime))
                value.append(count)
    
    # Esquema de cores aprimorado - cores mais contrastantes
    node_colors = (
        ['#FF6B6B', '#FF8E53', '#FF6B9D', '#C44569', '#F8B500', 
         '#6C5CE7', '#A29BFE', '#FD79A8', '#E17055', '#00B894'] +  # Vínculos
        ['#00CEC9', '#0984E3', '#6C5CE7', '#A29BFE', '#FD79A8', 
         '#E17055', '#00B894', '#FDCB6E', '#E84393', '#74B9FF']    # Crimes
    )
    
    # Garantir que temos cores suficientes
    while len(node_colors) < len(all_labels):
        node_colors.extend(node_colors)
    
    node_colors = node_colors[:len(all_labels)]
    
    # Criar o diagrama Sankey
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="black", width=1.5),
            label=[f"<b>{label}</b>" for label in all_labels],  # Labels em negrito
            color=node_colors,
            hovertemplate='%{label}<br>Total: %{value}<extra></extra>'
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=['rgba(0, 123, 191, 0.3)' for _ in value],  # Azul semi-transparente
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

def create_seasonality_analysis(df):
    """
    Cria análise de sazonalidade avançada com três gráficos.
    """
    if 'data_denuncia' not in df.columns:
        return None, None, None
    
    df_clean = df.dropna(subset=['data_denuncia']).copy()
    if len(df_clean) == 0:
        return None, None, None
    
    # Adicionar colunas de mês, ano e dia da semana
    df_clean['Month'] = df_clean['data_denuncia'].dt.month
    df_clean['Year'] = df_clean['data_denuncia'].dt.year
    df_clean['Weekday'] = df_clean['data_denuncia'].dt.dayofweek  # 0=segunda, 1=terça, ..., 6=domingo
    
    month_names = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
                   'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']
    
    # Gráfico 1: Sazonalidade mensal (média por mês)
    monthly_counts = df_clean.groupby(['Year', 'Month']).size().reset_index(name='casos')
    monthly_total = monthly_counts.groupby('Month')['casos'].mean()
    mean_cases = monthly_total.mean()
    std_cases = monthly_total.std()
    upper_limit = mean_cases + std_cases
    
    colors = ['red' if cases > upper_limit else 'lightblue' for cases in monthly_total]
    
    fig_seasonal = go.Figure()
    
    fig_seasonal.add_trace(go.Bar(
        x=month_names,
        y=monthly_total.values,
        marker_color=colors,
        text=[f"{val:.1f}" for val in monthly_total.values],
        textposition='outside',
        name='Média de Casos por Mês'
    ))
    
    fig_seasonal.add_hline(
        y=mean_cases, line_dash="dash", line_color="green",
        annotation_text=f"Média: {mean_cases:.0f}", annotation_position="top right"
    )
    
    fig_seasonal.add_hline(
        y=upper_limit, line_dash="dot", line_color="red",
        annotation_text=f"Alto: {upper_limit:.0f}", annotation_position="top right"
    )
    
    fig_seasonal.update_layout(
        title="📊 Análise de Sazonalidade - Média de Casos por Mês",
        xaxis_title="Mês", yaxis_title="Média de Casos",
        height=500, showlegend=False, template='plotly_white'
    )
    
    # Gráfico 2: Comparação ano a ano
    yearly_monthly = df_clean.groupby(['Year', 'Month']).size().reset_index(name='casos')
    
    fig_yearly = go.Figure()
    
    colors_yearly = px.colors.qualitative.Set1
    for i, year in enumerate(sorted(yearly_monthly['Year'].unique())):
        year_data = yearly_monthly[yearly_monthly['Year'] == year]
        fig_yearly.add_trace(go.Scatter(
            x=[month_names[m-1] for m in year_data['Month']],
            y=year_data['casos'],
            mode='lines+markers',
            name=str(year),
            line=dict(color=colors_yearly[i % len(colors_yearly)], width=3),
            marker=dict(size=8)
        ))
    
    fig_yearly.update_layout(
        title="📈 Comparação Ano a Ano - Padrões Mensais",
        xaxis_title="Mês",
        yaxis_title="Número de Casos",
        height=500,
        template='plotly_white',
        legend=dict(title="Ano")
    )
    
    # <<< LINHA ADICIONADA PARA CORRIGIR O EIXO X >>>
    fig_yearly.update_xaxes(categoryorder='array', categoryarray=month_names)
    
    # Gráfico 3: Tendência por dia da semana (média por dia)
    df_clean['YearMonth'] = df_clean['data_denuncia'].dt.to_period('M')
    weekday_counts = df_clean.groupby(['YearMonth', 'Weekday']).size().reset_index(name='casos')
    weekday_total = weekday_counts.groupby('Weekday')['casos'].mean()
    weekday_mean = weekday_total.mean()
    weekday_std = weekday_total.std()
    weekday_upper_limit = weekday_mean + weekday_std
    
    weekday_names = ['Segunda-feira', 'Terça-feira', 'Quarta-feira', 'Quinta-feira', 
                     'Sexta-feira', 'Sábado', 'Domingo']
    
    weekday_colors = ['red' if cases > weekday_upper_limit else 'lightcoral' for cases in weekday_total]
    
    fig_weekday = go.Figure()
    
    fig_weekday.add_trace(go.Bar(
        x=weekday_names,
        y=weekday_total.values,
        marker_color=weekday_colors,
        text=[f"{val:.1f}" for val in weekday_total.values],
        textposition='outside',
        name='Média de Casos por Dia da Semana'
    ))
    
    fig_weekday.add_hline(
        y=weekday_mean, line_dash="dash", line_color="green",
        annotation_text=f"Média: {weekday_mean:.0f}", annotation_position="top right"
    )
    
    fig_weekday.add_hline(
        y=weekday_upper_limit, line_dash="dot", line_color="red",
        annotation_text=f"Alto: {weekday_upper_limit:.0f}", annotation_position="top right"
    )
    
    fig_weekday.update_layout(
        title="📅 Tendência por Dia da Semana - Média de Casos",
        xaxis_title="Dia da Semana", yaxis_title="Média de Casos",
        height=500, showlegend=False, template='plotly_white',
        xaxis=dict(tickangle=45)
    )
    
    return fig_seasonal, fig_yearly, fig_weekday

def create_crime_category_boxplot(df):
    """
    Cria box plot da distribuição de idade por categoria de crime.
    """
    if 'idade_mulher' not in df.columns or 'categoria_crime' not in df.columns:
        return None
    
    df_clean = df.dropna(subset=['idade_mulher', 'categoria_crime'])
    if len(df_clean) == 0:
        return None
    
    fig = px.box(
        df_clean,
        x='categoria_crime',
        y='idade_mulher',
        title='📊 Distribuição de Idade das Vítimas por Categoria de Crime',
        labels={'categoria_crime': 'Categoria de Crime', 'idade_mulher': 'Idade da Vítima'},
        color='categoria_crime',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        height=600,
        xaxis_tickangle=45,
        template='plotly_white',
        showlegend=False
    )
    
    # Adicionar estatísticas na legenda
    stats_text = []
    for categoria in df_clean['categoria_crime'].unique():
        data = df_clean[df_clean['categoria_crime'] == categoria]['idade_mulher']
        median_age = data.median()
        std_age = data.std()
        stats_text.append(f"{categoria}: μ={median_age:.1f}, σ={std_age:.1f}")
    
    fig.add_annotation(
        text="<br>".join(stats_text),
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10)
    )
    
    return fig

def create_vinculo_crime_heatmap(df):
    """
    Cria heatmap cruzando vínculo com tipo de crime.
    """
    if 'vinculo' not in df.columns or 'titulo_fato' not in df.columns:
        return None
    
    # Limpar dados
    df_clean = df.dropna(subset=['vinculo', 'titulo_fato'])
    df_clean = df_clean[
        (df_clean['vinculo'] != 'Não Informado') & 
        (df_clean['titulo_fato'] != 'Não Informado')
    ]
    
    if len(df_clean) == 0:
        return None
    
    # Criar tabulação cruzada
    crosstab = pd.crosstab(df_clean['vinculo'], df_clean['titulo_fato'])
    
    # Filtrar para mostrar apenas os principais tipos
    top_vinculos = df_clean['vinculo'].value_counts().head(10).index
    top_crimes = df_clean['titulo_fato'].value_counts().head(10).index
    
    crosstab_filtered = crosstab.loc[
        crosstab.index.intersection(top_vinculos),
        crosstab.columns.intersection(top_crimes)
    ]
    
    if crosstab_filtered.empty:
        return None
    
    fig = px.imshow(
        crosstab_filtered.values,
        x=crosstab_filtered.columns,
        y=crosstab_filtered.index,
        title='🔥 Mapa de Calor: Tipo de Vínculo vs Tipo de Crime',
        labels=dict(x="Tipo de Crime", y="Vínculo", color="Número de Casos"),
        color_continuous_scale='Reds',
        aspect='auto',
        text_auto=True
    )
    
    fig.update_layout(
        height=600,
        xaxis_tickangle=45,
        template='plotly_white'
    )
    
    return fig

def create_age_difference_histogram(df):
    """
    Cria histograma da diferença de idade entre agressor e vítima.
    """
    if 'idade_agressor' not in df.columns or 'idade_mulher' not in df.columns:
        return None
    
    df_clean = df.dropna(subset=['idade_agressor', 'idade_mulher']).copy()
    if len(df_clean) == 0:
        return None
    
    # Calcular diferença de idade
    df_clean['diferenca_idade'] = df_clean['idade_agressor'] - df_clean['idade_mulher']
    
    # Filtrar outliers extremos
    q1 = df_clean['diferenca_idade'].quantile(0.01)
    q99 = df_clean['diferenca_idade'].quantile(0.99)
    df_filtered = df_clean[(df_clean['diferenca_idade'] >= q1) & (df_clean['diferenca_idade'] <= q99)]
    
    mean_diff = df_filtered['diferenca_idade'].mean()
    
    fig = px.histogram(
        df_filtered,
        x='diferenca_idade',
        nbins=50,
        title='📊 Distribuição da Diferença de Idade (Agressor - Vítima)',
        labels={'diferenca_idade': 'Diferença de Idade (anos)', 'count': 'Frequência'},
        color_discrete_sequence=['lightcoral']
    )
    
    # Adicionar linha da média
    fig.add_vline(
        x=mean_diff,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Média: {mean_diff:.1f} anos",
        annotation_position="top"
    )
    
    fig.update_layout(
        height=500,
        template='plotly_white',
        showlegend=False
    )
    
    return fig

def create_choropleth_map(df):
    """
    Cria mapa coroplético usando geobr e geopandas.
    """
    if 'municipio_corrigido' not in df.columns:
        return None
    
    try:
        # Carregar dados geográficos de SC
        sc_municipalities = geobr.read_municipality(code_muni="SC", year=2020)
        
        # Contar casos por município
        muni_counts = df['municipio_corrigido'].value_counts().reset_index()
        muni_counts.columns = ['municipio', 'casos']
        
        # Normalizar nomes dos municípios para melhor matching
        sc_municipalities['name_muni_clean'] = sc_municipalities['name_muni'].str.upper().str.strip()
        muni_counts['municipio_clean'] = muni_counts['municipio'].str.upper().str.strip()
        
        # Merge dos dados
        merged_data = sc_municipalities.merge(
            muni_counts, 
            left_on='name_muni_clean', 
            right_on='municipio_clean', 
            how='left'
        )
        merged_data['casos'] = merged_data['casos'].fillna(0)
        
        # Criar mapa base
        center_lat, center_lon = -27.2423, -50.2189
        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=7, 
            tiles='OpenStreetMap'
        )
        
        # Adicionar choropleth
        folium.Choropleth(
            geo_data=merged_data.to_json(),
            name='choropleth',
            data=merged_data,
            columns=['code_muni', 'casos'],
            key_on='feature.properties.code_muni',
            fill_color='YlOrRd',
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name='Número de Casos'
        ).add_to(m)
        
        # Adicionar tooltips
        for idx, row in merged_data.iterrows():
            if row['casos'] > 0:
                folium.Marker(
                    [row.geometry.centroid.y, row.geometry.centroid.x],
                    popup=f"{row['name_muni']}: {row['casos']} casos",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        return m
        
    except Exception as e:
        st.warning(f"Erro ao criar mapa coroplético: {str(e)}. Usando mapa simplificado.")
        return create_simplified_map(df)

def create_simplified_map(df):
    """
    Cria mapa simplificado em caso de erro no mapa coroplético.
    """
    if 'municipio_corrigido' not in df.columns:
        return None
    
    # Agregar dados por município
    muni_counts = df['municipio_corrigido'].value_counts().reset_index()
    muni_counts.columns = ['Município', 'Casos']
    
    # Criar mapa base centrado em SC
    center_lat, center_lon = -27.2423, -50.2189
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=7, 
        tiles='OpenStreetMap'
    )
    
    # Adicionar heatmap com dados aproximados
    heat_data = []
    max_cases = muni_counts['Casos'].max()
    
    for _, row in muni_counts.head(20).iterrows():
        # Coordenadas aproximadas (em produção, usar geocoding real)
        lat = center_lat + np.random.uniform(-2, 2)
        lon = center_lon + np.random.uniform(-3, 3)
        weight = row['Casos'] / max_cases
        heat_data.append([lat, lon, weight])
    
    # Adicionar heatmap
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
    
    return m

def create_time_series(df):
    """
    Cria gráfico de série temporal de denúncias.
    """
    # Agrupa por mês/ano
    df_monthly = df.set_index('data_denuncia').resample('M').size().reset_index()
    df_monthly.columns = ['data', 'casos']
    
    fig = px.line(
        df_monthly, 
        x='data', 
        y='casos',
        title='📈 Série Temporal de Denúncias por Mês',
        labels={'casos': 'Número de Casos', 'data': 'Data'},
        template='plotly_white'
    )
    
    fig.update_traces(line_color='#667eea', line_width=3)
    fig.update_layout(
        height=400,
        title_font_size=16,
        showlegend=False
    )
    
    return fig

def create_top_municipalities(df):
    """
    Cria gráfico dos top 10 municípios com mais ocorrências.
    """
    if 'municipio_corrigido' not in df.columns:
        return None
        
    top_municipios = df['municipio_corrigido'].value_counts().head(10)
    
    fig = px.bar(
        x=top_municipios.values,
        y=top_municipios.index,
        orientation='h',
        title='🏙️ Top 10 Municípios com Mais Ocorrências',
        labels={'x': 'Número de Casos', 'y': 'Município'},
        template='plotly_white',
        color=top_municipios.values,
        color_continuous_scale='inferno'
    )
    
    fig.update_layout(
        height=400,
        title_font_size=16,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_vinculo_distribution(df):
    """
    Cria gráfico de distribuição por vínculo agressor-vítima.
    """
    if 'vinculo' not in df.columns:
        return None
        
    vinculo_counts = df['vinculo'].value_counts()
    
    fig = px.pie(
        values=vinculo_counts.values,
        names=vinculo_counts.index,
        title='💔 Distribuição por Vínculo Agressor-Vítima',
        template='plotly_white',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        height=400,
        title_font_size=16,
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
    )
    
    return fig

def create_demographic_pyramid(df):
    """
    Cria pirâmide etária comparando vítimas e agressores.
    """
    if 'idade_mulher' not in df.columns or 'idade_agressor' not in df.columns:
        return None
    
    # Preparar dados de idade
    victim_ages = df['idade_mulher']#.dropna()
    aggressor_ages = df['idade_agressor']#.dropna()
    
    # Criar bins de idade
    bins = range(0, 100, 5)
    victim_hist, _ = np.histogram(victim_ages, bins=bins)
    aggressor_hist, _ = np.histogram(aggressor_ages, bins=bins)
    
    # Criar labels das faixas etárias
    age_labels = [f"{i}-{i+4}" for i in bins[:-1]]
    
    fig = go.Figure()
    
    # Vítimas (lado esquerdo, valores negativos)
    fig.add_trace(go.Bar(
        y=age_labels,
        x=-victim_hist,
        name='Vítimas',
        orientation='h',
        marker=dict(color='#FF6B6B'),
        hovertemplate='Faixa Etária: %{y}<br>Vítimas: %{x}<extra></extra>'
    ))
    
    # Agressores (lado direito, valores positivos)
    fig.add_trace(go.Bar(
        y=age_labels,
        x=aggressor_hist,
        name='Agressores',
        orientation='h',
        marker=dict(color='#4ECDC4'),
        hovertemplate='Faixa Etária: %{y}<br>Agressores: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title="📊 Pirâmide Etária: Vítimas vs Agressores",
        title_font_size=16,
        xaxis_title="Número de Casos",
        yaxis_title="Faixa Etária",
        height=600,
        barmode='overlay',
        showlegend=True,
        template='plotly_white'
    )
    
    # Adicionar linha central
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    return fig

def create_education_correlation_heatmap(df):
    """
    Cria heatmap cruzando escolaridade da vítima com a do agressor.
    """
    if 'nome_escolaridade_mulher' not in df.columns or 'nome_escolaridade_agressor' not in df.columns:
        return None
    
    # Limpar dados
    df_clean = df.dropna(subset=['nome_escolaridade_mulher', 'nome_escolaridade_agressor'])
    df_clean = df_clean[
        (df_clean['nome_escolaridade_mulher'] != 'Não Informado') & 
        (df_clean['nome_escolaridade_agressor'] != 'Não Informado')
    ]
    
    if len(df_clean) == 0:
        return None
    
    # Criar tabulação cruzada
    crosstab = pd.crosstab(
        df_clean['nome_escolaridade_mulher'], 
        df_clean['nome_escolaridade_agressor']
    )
    
    # Criar heatmap
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

def create_education_comparison_enhanced(df):
    """
    Análise aprimorada de escolaridade com ordenação adequada
    """
    if 'nome_escolaridade_mulher' not in df.columns or 'nome_escolaridade_agressor' not in df.columns:
        return None
    
    # Definir ordem de educação do menor para o maior nível
    education_order = [
            'Não Alfabetizado', 'Semialfabetizado',
            'Ensino fundamental incompleto', 'Ensino fundamental completo',
            'Ensino médio incompleto', 'Ensino Médio Completo',
            'Superior incompleto', 'Superior (cursando)', 'Superior completo',
            'Pós Graduação', 'Mestrado', 'Doutorado'
        ]
    
    
    # Limpar dados
    victim_edu = df['nome_escolaridade_mulher'].value_counts()
    aggressor_edu = df['nome_escolaridade_agressor'].value_counts()
    
    # Obter níveis de educação comuns e ordená-los
    all_education_levels = set(victim_edu.index) | set(aggressor_edu.index)
    all_education_levels = [edu for edu in all_education_levels if edu != 'Não Informado']
    
    # Ordenar níveis de educação
    ordered_education = []
    for level in education_order:
        if level in all_education_levels:
            ordered_education.append(level)
    
    # Adicionar níveis restantes não na ordem predefinida
    for level in all_education_levels:
        if level not in ordered_education:
            ordered_education.append(level)
    
    if len(ordered_education) == 0:
        return None
    
    # Criar dataframe de comparação
    education_df = pd.DataFrame({
        'Vítimas': [victim_edu.get(level, 0) for level in ordered_education],
        'Agressores': [aggressor_edu.get(level, 0) for level in ordered_education]
    }, index=ordered_education)
    
    # Calcular percentuais
    education_df['Vítimas_Pct'] = education_df['Vítimas'] / education_df['Vítimas'].sum() * 100
    education_df['Agressores_Pct'] = education_df['Agressores'] / education_df['Agressores'].sum() * 100
    
    # Criar subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribuição Absoluta', 'Distribuição Percentual'),
        specs=[[{'secondary_y': False}, {'secondary_y': False}]]
    )
    
    # Números absolutos
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
    
    # Percentuais
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

def main():
    """
    Função principal do dashboard.
    """
    # Título principal
    st.markdown('<h1 class="main-header">📊 Dashboard Avançado de Análise de Violência contra a Mulher - SC</h1>', 
                unsafe_allow_html=True)
    
    # Carrega os dados
    with st.spinner('🔄 Carregando dados...'):
        df = load_data()
    
    if df is None:
        st.stop()
    
    # Aplicar categorização de crimes
    df = categorize_crime(df)
    
    # Sidebar com filtros
    st.sidebar.header('🔍 Filtros Avançados')
    
    # Filtro de período
    min_date = df[df['data_denuncia'].dt.year == 2020]['data_denuncia'].min().date()
    max_date = df[df['data_denuncia'].dt.year == 2025]['data_denuncia'].max().date()
    
    data_inicio, data_fim = st.sidebar.date_input(
        '📅 Período de Análise',
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filtro de município
    if 'municipio_corrigido' in df.columns:
        municipios_disponiveis = sorted([m for m in df['municipio_corrigido'].unique() if m != 'Não Informado'])
        municipios_selecionados = st.sidebar.multiselect(
            '🏙️ Municípios',
            options=municipios_disponiveis,
            default=municipios_disponiveis  # Limite inicial para performance
        )
    else:
        municipios_selecionados = []
    
    # Filtro de categoria de crime
    categorias_disponiveis = sorted(df['categoria_crime'].unique())
    categorias_selecionadas = st.sidebar.multiselect(
        '🔍 Categoria de Crime',
        options=categorias_disponiveis,
        default=categorias_disponiveis
    )
    
    # Aplicar filtros
    df_filtrado = df.copy()
    
    # Filtro de data
    df_filtrado = df_filtrado[
        (df_filtrado['data_denuncia'].dt.date >= data_inicio) & 
        (df_filtrado['data_denuncia'].dt.date <= data_fim)
    ]
    
    # Filtro de município
    if municipios_selecionados and 'municipio_corrigido' in df.columns:
        df_filtrado = df_filtrado[df_filtrado['municipio_corrigido'].isin(municipios_selecionados)]
    
    # Filtro de categoria
    if categorias_selecionadas:
        df_filtrado = df_filtrado[df_filtrado['categoria_crime'].isin(categorias_selecionadas)]
    
    # Verificar se há dados após filtros
    if len(df_filtrado) == 0:
        st.warning('⚠️ Nenhum dado encontrado com os filtros selecionados.')
        return
    
    # Mostrar informações dos filtros
    st.sidebar.markdown('---')
    st.sidebar.metric("📊 Registros Filtrados", f"{len(df_filtrado):,}")
    
    # Overview com métricas
    st.markdown('<div class="success-box"><h3>📊 Visão Geral do Dataset</h3></div>', 
                unsafe_allow_html=True)
    
    total_denuncias, vitimas_unicas, agressores_unicos = calculate_kpis(df_filtrado)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("📋 Total de Casos", f"{total_denuncias:,}")
    
    with col2:
        st.metric("👥 Vítimas Únicas", vitimas_unicas if vitimas_unicas != 'N/A' else vitimas_unicas)
    
    with col3:
        st.metric("👤 Agressores Únicos", agressores_unicos if agressores_unicos != 'N/A' else agressores_unicos)
    
    with col4:
        if 'municipio_corrigido' in df_filtrado.columns:
            st.metric("🏙️ Municípios", df_filtrado['municipio_corrigido'].nunique())
        else:
            st.metric("🏙️ Municípios", "N/A")
    
    with col5:
        periodo_dias = (data_fim - data_inicio).days + 1
        st.metric("📅 Período (dias)", f"{periodo_dias}")
    
    # Estrutura de abas - MODIFICADA PARA INCLUIR A NOVA ABA
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        '🏠 Análise Geral', 
        '👥 Demografia', 
        '📅 Padrões Temporais', 
        '🔍 Análise de Crimes',
        '🗺️ Análise Geográfica',
        '👥 Análise Étnico-Racial'
    ])
    
    with tab1:
        st.header('🎯 Análise Geral')
        
        # Série temporal
        fig_temporal = create_time_series(df_filtrado)
        if fig_temporal:
            st.plotly_chart(fig_temporal, use_container_width=True)
        
        # Diagrama de Sankey
        sankey_fig = create_sankey_diagram(df_filtrado)
        if sankey_fig:
            st.plotly_chart(sankey_fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
                <h4>🌊 Interpretação do Diagrama de Sankey</h4>
                <p>• <strong>Lado Esquerdo (Azul):</strong> Top 10 tipos de vínculo entre agressor e vítima</p>
                <p>• <strong>Lado Direito (Vermelho):</strong> Top 10 tipos de crime mais denunciados</p>
                <p>• <strong>Fluxos:</strong> A espessura das conexões indica o volume de casos</p>
                <p>• <strong>Insights:</strong> Identifica quais vínculos estão associados a quais tipos de crime</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Dados insuficientes para criar o diagrama de Sankey")
        
        # Layout em colunas
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição por vínculo
            fig_vinculo = create_vinculo_distribution(df_filtrado)
            if fig_vinculo:
                st.plotly_chart(fig_vinculo, use_container_width=True)
        
        with col2:
            # Top municípios
            fig_municipios = create_top_municipalities(df_filtrado)
            if fig_municipios:
                st.plotly_chart(fig_municipios, use_container_width=True)
    
    with tab2:
        st.header('👥 Análise Demográfica')
        
        # Pirâmide etária
        pyramid_fig = create_demographic_pyramid(df_filtrado)
        if pyramid_fig:
            st.plotly_chart(pyramid_fig, use_container_width=True)
        else:
            st.info("Dados de idade insuficientes para a pirâmide demográfica")
        
        # Layout em colunas para outros gráficos demográficos
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de comparação de escolaridade aprimorado
            edu_comparison = create_education_comparison_enhanced(df_filtrado)
            if edu_comparison:
                st.plotly_chart(edu_comparison, use_container_width=True)
                
                st.markdown("""
                <div class="insight-box">
                    <h4>📚 Análise de Escolaridade</h4>
                    <p>• <strong>Lado Esquerdo:</strong> Distribuição absoluta por nível educacional</p>
                    <p>• <strong>Lado Direito:</strong> Distribuição percentual comparativa</p>
                    <p>• <strong>Ordenação:</strong> Do menor ao maior nível de escolaridade</p>
                    <p>• <strong>Insights:</strong> Compare perfis educacionais entre vítimas e agressores</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Dados de escolaridade insuficientes para comparação")
        
        with col2:
            # Histograma de diferença de idade
            age_diff_hist = create_age_difference_histogram(df_filtrado)
            if age_diff_hist:
                st.plotly_chart(age_diff_hist, use_container_width=True)
            else:
                st.info("Dados de idade insuficientes para histograma")
        
        # Heatmap de escolaridade (linha separada para mais espaço)
        edu_heatmap = create_education_correlation_heatmap(df_filtrado)
        if edu_heatmap:
            st.plotly_chart(edu_heatmap, use_container_width=True)
        else:
            st.info("Dados de escolaridade insuficientes para heatmap")
    
    with tab3:
        st.header('📅 Padrões Temporais')
        
        # Análise de sazonalidade

        seasonal_fig, yearly_fig, weekday_fig = create_seasonality_analysis(df_filtrado)
        
        if seasonal_fig:
            st.plotly_chart(seasonal_fig, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
                <h4>📊 Interpretação da Sazonalidade</h4>
                <p>• Barras vermelhas: Meses com casos acima da média + 1 desvio padrão</p>
                <p>• Barras azuis: Meses com padrão normal de casos</p>
                <p>• Linha verde tracejada: Média mensal de casos</p>
                <p>• Linha vermelha pontilhada: Limite superior (Média + DP)</p>
            </div>
            """, unsafe_allow_html=True)
        
        if weekday_fig:
            st.plotly_chart(weekday_fig, use_container_width=True)
            
        if yearly_fig:
            st.plotly_chart(yearly_fig, use_container_width=True)
            
            st.markdown("""
            <div class="warning-box">
                <h4>📈 Comparação Anual</h4>
                <p>• Cada linha representa um ano diferente</p>
                <p>• Compare padrões sazonais entre anos</p>
                <p>• Identifique tendências de crescimento ou redução</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:
        st.header('🔍 Análise de Crimes')
        
        # Box plot por categoria de crime
        crime_boxplot = create_crime_category_boxplot(df_filtrado)
        if crime_boxplot:
            st.plotly_chart(crime_boxplot, use_container_width=True)
            
            
            st.markdown("""
            <div class="crime-category-box">
                <h4>📊 Categorização de Crimes</h4>
                <p>• <strong>Violência Física:</strong> Lesão Corporal</p>
                <p>• <strong>Violência Psicológica:</strong> Ameaça, Constrangimento ilegal, Violência Psicológica</p>
                <p>• <strong>Violência Moral:</strong> Injúria, Difamação, Calúnia</p>
                <p>• <strong>Violência Sexual:</strong> Estupro, Sedução, Assédio Sexual</p>
                <p>• <strong>Violência Patrimonial:</strong> Dano, Furto, Roubo, Apropriação Indébita</p>
                <p>• <strong>Violência de Múltipla Dimensões:</strong> Outro, Sequestro e Carcere Privado</>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Dados insuficientes para análise por categoria")
        
        # Heatmap vínculo vs crime
        vinculo_crime_heatmap = create_vinculo_crime_heatmap(df_filtrado)
        if vinculo_crime_heatmap:
            st.plotly_chart(vinculo_crime_heatmap, use_container_width=True)
            
            st.markdown("""
            <div class="insight-box">
                <h4>🔥 Mapa de Calor Vínculo vs Crime</h4>
                <p>• Cores mais intensas indicam maior concentração de casos</p>
                <p>• Identifique padrões entre tipos de relacionamento e crimes</p>
                <p>• Ajuda na criação de políticas públicas direcionadas</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Dados insuficientes para criar mapa de calor")
    
    with tab5:
        st.header('🗺️ Análise Geográfica')
        
        # Mapa coroplético por municípios
        st.subheader('🌍 Mapa Coroplético de Santa Catarina - Municípios')
        
        choropleth_map = create_choropleth_map(df_filtrado)
        if choropleth_map:
            folium_html = choropleth_map._repr_html_()
            components.html(folium_html, height=600)
            
            st.markdown("""
            <div class="success-box">
                <h4>🗺️ Interpretação do Mapa Coroplético</h4>
                <p>• Cores mais intensas: Maior concentração de casos</p>
                <p>• Fronteiras reais dos municípios de Santa Catarina</p>
                <p>• Clique nos marcadores para ver detalhes por município</p>
                <p>• Use os controles do mapa para navegar e fazer zoom</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Mapa não disponível")
        
        # NOVO: Mapa de mesorregiões
        st.subheader('🗺️ Mapa de Casos por Mesorregião')
        mesoregion_map = create_mesoregion_choropleth_map(df_filtrado)
        if mesoregion_map:
            folium_html = mesoregion_map._repr_html_()
            components.html(folium_html, height=500)
        else:
            st.info("Dados insuficientes para o mapa de mesorregiões.")
        
        # Top municípios em tabela
        if 'municipio_corrigido' in df_filtrado.columns:
            st.subheader('🏆 Top 15 Municípios')
            
            muni_counts = df_filtrado['municipio_corrigido'].value_counts().head(15)
            muni_df = pd.DataFrame({
                'Ranking': range(1, len(muni_counts) + 1),
                'Município': muni_counts.index,
                'Casos': muni_counts.values,
                'Porcentagem': (muni_counts.values / len(df_filtrado) * 100).round(2)
            })
            st.dataframe(muni_df, use_container_width=True)
    
    # NOVA ABA: Análise Étnico-Racial
    with tab6:
        st.header('👥 Análise Étnico-Racial')
        
        if 'cor_raca_vitima' in df_filtrado.columns:
            # Gráfico de distribuição
            fig_race_dist = create_race_distribution_chart(df_filtrado)
            if fig_race_dist:
                st.plotly_chart(fig_race_dist, use_container_width=True)

            # Heatmap Raça vs Crime
            fig_race_crime = create_race_crime_heatmap(df_filtrado)
            if fig_race_crime:
                st.plotly_chart(fig_race_crime, use_container_width=True)
                
                st.markdown("""
                <div class="insight-box">
                    <h4>🔥 Análise Étnico-Racial por Tipo de Crime</h4>
                    <p>• <strong>Mapa de Calor:</strong> Visualiza a intersecção entre raça/cor e categorias de crime</p>
                    <p>• <strong>Cores Intensas:</strong> Indicam maior concentração de casos</p>
                    <p>• <strong>Padrões:</strong> Identifica desigualdades e vulnerabilidades específicas</p>
                    <p>• <strong>Políticas Públicas:</strong> Dados fundamentais para ações afirmativas e proteção</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Dados insuficientes para criar mapa de calor étnico-racial")
        else:
            st.warning("Dados de raça/cor não disponíveis para análise.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>📊 <strong>Dashboard Avançado de Violência contra a Mulher - Santa Catarina</strong></p>
        <p>Desenvolvido com ❤️ usando Streamlit, Plotly, Folium e GeoBR</p>
        <p>🔒 Dados tratados com responsabilidade e confidencialidade</p>
        <p>📈 Análises: Sazonalidade, Demografia, Categorização, Padrões Temporais, Geográficos e Étnico-Raciais</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
