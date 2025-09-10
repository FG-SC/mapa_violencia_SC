# config_utils.py
"""
Configuration and utility functions for the Violence Analysis Dashboard
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Tuple, Optional
import plotly.colors as pc

# Color schemes for different chart types
COLOR_SCHEMES = {
    'primary': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'],
    'sequential': ['#FFF5F0', '#FEE0D2', '#FCBBA1', '#FC9272', '#FB6A4A', '#EF3B2C', '#CB181D', '#A50F15', '#67000D'],
    'diverging': ['#8E0152', '#C51B7D', '#DE77AE', '#F1B6DA', '#FDE0EF', '#E6F5D0', '#B8E186', '#7FBC41', '#4D9221', '#276419'],
    'qualitative': ['#E8743B', '#19A979', '#ED4A7B', '#945ECF', '#13A4B4', '#525DF4', '#BF399E', '#6C8893', '#EE6A24', '#009432']
}

# Santa Catarina municipalities by meso-region (complete mapping)
SC_MESOREGIONS_DETAILED = {
    'Oeste Catarinense': [
        # São Miguel do Oeste microregion
        'Anchieta', 'Bandeirante', 'Barra Bonita', 'Belmonte', 'Descanso', 
        'Dionísio Cerqueira', 'Guaraciaba', 'Guarujá do Sul', 'Iporã do Oeste', 
        'Itapiranga', 'Mondaí', 'Palma Sola', 'Paraíso', 'Princesa', 'Riqueza', 
        'Romelândia', 'Santa Helena', 'São João do Oeste', 'São José do Cedro', 
        'São Miguel do Oeste', 'Tunápolis',
        # Chapecó microregion
        'Águas de Chapecó', 'Águas Frias', 'Bom Jesus do Oeste', 'Caibi', 
        'Campo Erê', 'Caxambu do Sul', 'Chapecó', 'Cordilheira Alta', 
        'Coronel Freitas', 'Cunha Porã', 'Cunhataí', 'Flor do Sertão', 
        'Formosa do Sul', 'Guatambu', 'Iraceminha', 'Irati', 'Jardinópolis', 
        'Maravilha', 'Modelo', 'Nova Erechim', 'Nova Itaberaba', 'Novo Horizonte', 
        'Palmitos', 'Pinhalzinho', 'Planalto Alegre', 'Quilombo', 'Saltinho', 
        'Santa Terezinha do Progresso', 'Santiago do Sul', 'São Bernardino', 
        'São Carlos', 'São Lourenço do Oeste', 'São Miguel da Boa Vista', 
        'Saudades', 'Serra Alta', 'Sul Brasil', 'Tigrinhos', 'União do Oeste',
        # Xanxerê microregion
        'Abelardo Luz', 'Bom Jesus', 'Coronel Martins', 'Entre Rios', 
        'Faxinal dos Guedes', 'Galvão', 'Ipuaçu', 'Jupiá', 'Lajeado Grande', 
        'Marema', 'Ouro Verde', 'Passos Maia', 'Ponte Serrada', 'São Domingos', 
        'Vargeão', 'Xanxerê', 'Xaxim',
        # Joaçaba microregion
        'Arroio Trinta', 'Caçador', 'Calmon', 'Capinzal', 'Catanduvas', 
        'Erval Velho', 'Fraiburgo', 'Herval d\'Oeste', 'Ibiam', 'Ibicaré', 
        'Iomerê', 'Jaborá', 'Joaçaba', 'Lacerdópolis', 'Lebon Régis', 'Luzerna', 
        'Macieira', 'Matos Costa', 'Ouro', 'Pinheiro Preto', 'Rio das Antas', 
        'Salto Veloso', 'Tangará', 'Treze Tílias', 'Vargem Bonita', 'Videira',
        # Concórdia microregion
        'Alto Bela Vista', 'Arabutã', 'Arvoredo', 'Concórdia', 'Ipira', 
        'Ipumirim', 'Irani', 'Itá', 'Lindóia do Sul', 'Paial', 'Peritiba', 
        'Piratuba', 'Presidente Castello Branco', 'Seara', 'Xavantina'
    ],
    'Serrana': [
        # Curitibanos microregion
        'Abdon Batista', 'Brunópolis', 'Campos Novos', 'Curitibanos', 
        'Frei Rogério', 'Monte Carlo', 'Ponte Alta', 'Ponte Alta do Norte', 
        'Santa Cecília', 'São Cristóvão do Sul', 'Vargem', 'Zortéa',
        # São Joaquim microregion
        'Anita Garibaldi', 'Bocaina do Sul', 'Bom Jardim da Serra', 'Bom Retiro', 
        'Campo Belo do Sul', 'Capão Alto', 'Celso Ramos', 'Cerro Negro', 
        'Correia Pinto', 'Lages', 'Otacílio Costa', 'Painel', 'Palmeira', 
        'Rio Rufino', 'São Joaquim', 'São José do Cerrito', 'Urubici', 'Urupema'
    ],
    'Vale do Itajaí': [
        # Alto Vale do Itajaí microregion
        'Agronômica', 'Aurora', 'Braço do Trombudo', 'Dona Emma', 'Ibirama', 
        'José Boiteux', 'Laurentino', 'Lontras', 'Mirim Doce', 'Pouso Redondo', 
        'Presidente Getúlio', 'Presidente Nereu', 'Rio do Campo', 'Rio do Oeste', 
        'Rio do Sul', 'Salete', 'Taió', 'Trombudo Central', 'Vitor Meireles', 
        'Witmarsum',
        # Blumenau microregion
        'Apiúna', 'Ascurra', 'Benedito Novo', 'Blumenau', 'Botuverá', 
        'Brusque', 'Doutor Pedrinho', 'Gaspar', 'Guabiruba', 'Indaial', 
        'Luiz Alves', 'Pomerode', 'Rio dos Cedros', 'Rodeio', 'Timbó',
        # Itajaí microregion
        'Balneário Camboriú', 'Balneário Piçarras', 'Barra Velha', 'Bombinhas', 
        'Camboriú', 'Ilhota', 'Itajaí', 'Itapema', 'Navegantes', 'Penha', 
        'Porto Belo', 'São João do Itaperiú',
        # Ituporanga microregion
        'Agrolândia', 'Atalanta', 'Chapadão do Lageado', 'Imbuia', 'Ituporanga', 
        'Petrolândia', 'Vidal Ramos'
    ],
    'Sul Catarinense': [
        # Tubarão microregion
        'Armazém', 'Braço do Norte', 'Capivari de Baixo', 'Garopaba', 
        'Grão-Pará', 'Gravatal', 'Imaruí', 'Imbituba', 'Jaguaruna', 'Laguna', 
        'Orleans', 'Pedras Grandes', 'Pescaria Brava', 'Rio Fortuna', 'Sangão', 
        'Santa Rosa de Lima', 'São Ludgero', 'São Martinho', 'Treze de Maio', 
        'Tubarão',
        # Criciúma microregion
        'Balneário Rincão', 'Cocal do Sul', 'Criciúma', 'Forquilhinha', 'Içara', 
        'Lauro Müller', 'Morro da Fumaça', 'Nova Veneza', 'Siderópolis', 
        'Treviso', 'Urussanga',
        # Araranguá microregion
        'Araranguá', 'Balneário Arroio do Silva', 'Balneário Gaivota', 'Ermo', 
        'Jacinto Machado', 'Maracajá', 'Meleiro', 'Morro Grande', 'Passo de Torres', 
        'Praia Grande', 'Santa Rosa do Sul', 'São João do Sul', 'Sombrio', 
        'Timbé do Sul', 'Turvo'
    ],
    'Grande Florianópolis': [
        # Tijucas microregion
        'Angelina', 'Canelinha', 'Leoberto Leal', 'Major Gercino', 'Nova Trento', 
        'São João Batista', 'Tijucas',
        # Florianópolis microregion
        'Antônio Carlos', 'Biguaçu', 'Florianópolis', 'Governador Celso Ramos', 
        'Palhoça', 'Paulo Lopes', 'Santo Amaro da Imperatriz', 'São José', 
        'São Pedro de Alcântara',
        # Tabuleiro microregion
        'Águas Mornas', 'Alfredo Wagner', 'Anitápolis', 'Rancho Queimado', 
        'São Bonifácio'
    ],
    'Norte Catarinense': [
        # Canoinhas microregion
        'Bela Vista do Toldo', 'Canoinhas', 'Irineópolis', 'Itaiópolis', 'Mafra', 
        'Major Vieira', 'Monte Castelo', 'Papanduva', 'Porto União', 
        'Santa Terezinha', 'Timbó Grande', 'Três Barras',
        # São Bento do Sul microregion
        'Campo Alegre', 'Rio Negrinho', 'São Bento do Sul',
        # Joinville microregion
        'Araquari', 'Balneário Barra do Sul', 'Corupá', 'Garuva', 'Guaramirim', 
        'Itapoá', 'Jaraguá do Sul', 'Joinville', 'Massaranduba', 
        'São Francisco do Sul', 'Schroeder'
    ]
}

# Crime severity mapping
CRIME_SEVERITY_MAP = {
    'Feminicídio': {'level': 5, 'color': '#8B0000', 'description': 'Extremamente Grave'},
    'Tentativa de Feminicídio': {'level': 5, 'color': '#8B0000', 'description': 'Extremamente Grave'},
    'Estupro': {'level': 5, 'color': '#DC143C', 'description': 'Extremamente Grave'},
    'Estupro de Vulnerável': {'level': 5, 'color': '#DC143C', 'description': 'Extremamente Grave'},
    'Lesão Corporal Grave': {'level': 4, 'color': '#FF4500', 'description': 'Grave'},
    'Lesão Corporal': {'level': 3, 'color': '#FF6347', 'description': 'Moderado'},
    'Ameaça': {'level': 3, 'color': '#FFA500', 'description': 'Moderado'},
    'Violência Psicológica': {'level': 3, 'color': '#FFD700', 'description': 'Moderado'},
    'Perseguição': {'level': 3, 'color': '#FFD700', 'description': 'Moderado'},
    'Injúria': {'level': 2, 'color': '#FFFF00', 'description': 'Leve'},
    'Difamação': {'level': 2, 'color': '#FFFF00', 'description': 'Leve'},
    'Calúnia': {'level': 2, 'color': '#FFFF00', 'description': 'Leve'},
    'Dano': {'level': 1, 'color': '#98FB98', 'description': 'Muito Leve'},
    'Perturbação da Tranquilidade': {'level': 1, 'color': '#98FB98', 'description': 'Muito Leve'}
}

# Education level standardization
EDUCATION_MAPPING = {
    'analfabeto': 'Analfabeto',
    'não alfabetizado': 'Analfabeto',
    'não alfabetizada': 'Analfabeto',
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
    'ensino superior completo': 'Superior Completo',
    'pós-graduação': 'Pós-graduação',
    'pós graduação': 'Pós-graduação',
    'mestrado': 'Pós-graduação',
    'doutorado': 'Pós-graduação'
}

# Relationship type standardization
RELATIONSHIP_MAPPING = {
    'matrimônio': 'Cônjuge/Matrimônio',
    'matrimonio': 'Cônjuge/Matrimônio',
    'casamento': 'Cônjuge/Matrimônio',
    'casado': 'Cônjuge/Matrimônio',
    'casada': 'Cônjuge/Matrimônio',
    'marido': 'Cônjuge/Matrimônio',
    'esposo': 'Cônjuge/Matrimônio',
    'união estável': 'União Estável',
    'uniao estavel': 'União Estável',
    'companheiro': 'União Estável',
    'companheira': 'União Estável',
    'namorado': 'Namorando',
    'namorada': 'Namorando',
    'namoro': 'Namorando',
    'ex-namorado': 'Ex-Namorado',
    'ex namorado': 'Ex-Namorado',
    'ex-marido': 'Ex-Cônjuge',
    'ex marido': 'Ex-Cônjuge',
    'ex-companheiro': 'Ex-Companheiro',
    'ex companheiro': 'Ex-Companheiro',
    'separado': 'Separado/Divorciado',
    'separada': 'Separado/Divorciado',
    'divorciado': 'Separado/Divorciado',
    'divorciada': 'Separado/Divorciado',
    'conhecido': 'Conhecido',
    'conhecida': 'Conhecido',
    'desconhecido': 'Desconhecido',
    'desconhecida': 'Desconhecido',
    'familiar': 'Familiar',
    'família': 'Familiar',
    'parente': 'Familiar',
    'pai': 'Familiar',
    'filho': 'Familiar',
    'irmão': 'Familiar',
    'tio': 'Familiar',
    'primo': 'Familiar',
    'amigo': 'Amigo/Conhecido',
    'amiga': 'Amigo/Conhecido',
    'vizinho': 'Vizinho',
    'vizinha': 'Vizinho',
    'colega': 'Colega/Trabalho',
    'trabalho': 'Colega/Trabalho',
    'solteiro': 'Sem Relacionamento',
    'solteira': 'Sem Relacionamento'
}

class DataProcessor:
    """Enhanced data processing utilities"""
    
    @staticmethod
    def get_mesoregion(municipality: str) -> str:
        """Map municipality to meso-region with improved fuzzy matching"""
        if pd.isna(municipality) or municipality in ['Não Informado', '']:
            return 'Não Informado'
        
        municipality_clean = str(municipality).strip().title()
        
        # Create a reverse mapping for faster lookup
        city_to_region = {}
        for region, cities in SC_MESOREGIONS_DETAILED.items():
            for city in cities:
                city_to_region[city.lower()] = region
        
        # Direct match (case-insensitive)
        if municipality_clean.lower() in city_to_region:
            return city_to_region[municipality_clean.lower()]
        
        # Fuzzy matching for partial matches
        municipality_lower = municipality_clean.lower()
        
        # Try to find partial matches
        for city_lower, region in city_to_region.items():
            # Check if municipality is contained in city name or vice versa
            if (municipality_lower in city_lower or 
                city_lower in municipality_lower or
                # Handle common variations
                municipality_lower.replace('ã', 'a').replace('ç', 'c').replace('õ', 'o') in city_lower or
                city_lower.replace('ã', 'a').replace('ç', 'c').replace('õ', 'o') in municipality_lower):
                return region
        
        # Handle common municipality name variations
        variations = {
            'florianopolis': 'florianópolis',
            'chapeco': 'chapecó',
            'blumenau': 'blumenau',
            'joinville': 'joinville',
            'criciuma': 'criciúma',
            'itajai': 'itajaí',
            'sao jose': 'são josé',
            'sao bento do sul': 'são bento do sul',
            'balneario camboriu': 'balneário camboriú'
        }
        
        for variation, correct in variations.items():
            if variation in municipality_lower:
                if correct.lower() in city_to_region:
                    return city_to_region[correct.lower()]
        
        return 'Outras Regiões'
    
    @staticmethod
    def standardize_education(education: str) -> str:
        """Standardize education levels"""
        if pd.isna(education) or education in ['Não Informado', '', 'nan']:
            return 'Não Informado'
        
        education_clean = str(education).strip().lower()
        
        for key, value in EDUCATION_MAPPING.items():
            if key in education_clean:
                return value
        
        return education_clean.title()
    
    @staticmethod
    def standardize_relationship(relationship: str) -> str:
        """Standardize relationship types"""
        if pd.isna(relationship) or relationship in ['Não Informado', '', 'nan']:
            return 'Não Informado'
        
        relationship_clean = str(relationship).strip().lower()
        
        for key, value in RELATIONSHIP_MAPPING.items():
            if key in relationship_clean:
                return value
        
        return relationship_clean.title()
    
    @staticmethod
    def get_crime_severity(crime_type: str) -> Dict:
        """Get crime severity information"""
        if pd.isna(crime_type) or crime_type in ['Não Informado', '', 'nan']:
            return {'level': 0, 'color': '#CCCCCC', 'description': 'Não Informado'}
        
        crime_clean = str(crime_type).strip().title()
        
        for crime, info in CRIME_SEVERITY_MAP.items():
            if crime.lower() in crime_clean.lower():
                return info
        
        return {'level': 2, 'color': '#FFA500', 'description': 'Moderado'}
    
    @staticmethod
    def calculate_risk_score(row: pd.Series) -> float:
        """Calculate risk score based on multiple factors"""
        score = 0.0
        
        # Crime severity
        if 'Tipo penal' in row:
            crime_info = DataProcessor.get_crime_severity(row['Tipo penal'])
            score += crime_info['level'] * 2
        
        # Age factors
        if 'idade - mulher' in row and pd.notna(row['idade - mulher']):
            age = row['idade - mulher']
            if age < 25 or age > 60:  # Higher risk for younger and older victims
                score += 1
        
        # Relationship factors
        if 'Vínculo' in row and pd.notna(row['Vínculo']):
            relationship = str(row['Vínculo']).lower()
            high_risk_relationships = ['ex-', 'separado', 'divorciado']
            if any(risk in relationship for risk in high_risk_relationships):
                score += 2
        
        # Normalize to 0-10 scale
        return min(score / 2, 10.0)

class VisualizationHelper:
    """Helper class for creating enhanced visualizations"""
    
    @staticmethod
    def get_color_scheme(chart_type: str, n_colors: int = None) -> List[str]:
        """Get appropriate color scheme for chart type"""
        schemes = COLOR_SCHEMES.get(chart_type, COLOR_SCHEMES['primary'])
        
        if n_colors:
            if n_colors <= len(schemes):
                return schemes[:n_colors]
            else:
                # Repeat colors if needed
                return (schemes * ((n_colors // len(schemes)) + 1))[:n_colors]
        
        return schemes
    
    @staticmethod
    def format_number(num: float, format_type: str = 'default') -> str:
        """Format numbers for display"""
        if pd.isna(num):
            return 'N/A'
        
        if format_type == 'percentage':
            return f"{num:.1f}%"
        elif format_type == 'currency':
            return f"R$ {num:,.2f}"
        elif format_type == 'compact':
            if num >= 1000000:
                return f"{num/1000000:.1f}M"
            elif num >= 1000:
                return f"{num/1000:.1f}K"
            else:
                return f"{num:.0f}"
        else:
            return f"{num:,.0f}" if num == int(num) else f"{num:,.2f}"
    
    @staticmethod
    def create_gradient_colors(start_color: str, end_color: str, n_steps: int) -> List[str]:
        """Create gradient colors between two colors"""
        # This is a simplified version - for production, use a proper color interpolation library
        colors = []
        for i in range(n_steps):
            ratio = i / (n_steps - 1) if n_steps > 1 else 0
            # Simple interpolation (in practice, you'd want RGB interpolation)
            colors.append(start_color if ratio < 0.5 else end_color)
        return colors

class StatisticalAnalyzer:
    """Statistical analysis utilities"""
    
    @staticmethod
    def calculate_demographic_stats(df: pd.DataFrame) -> Dict:
        """Calculate comprehensive demographic statistics"""
        stats = {}
        
        # Age statistics
        for age_col in ['idade - mulher', 'idade - agressor']:
            if age_col in df.columns:
                age_data = df[age_col].dropna()
                if len(age_data) > 0:
                    stats[age_col] = {
                        'mean': age_data.mean(),
                        'median': age_data.median(),
                        'mode': age_data.mode().iloc[0] if len(age_data.mode()) > 0 else None,
                        'std': age_data.std(),
                        'min': age_data.min(),
                        'max': age_data.max(),
                        'q1': age_data.quantile(0.25),
                        'q3': age_data.quantile(0.75),
                        'count': len(age_data)
                    }
        
        return stats
    
    @staticmethod
    def perform_correlation_analysis(df: pd.DataFrame) -> Dict:
        """Perform correlation analysis on numerical variables"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {}
        
        correlation_matrix = df[numerical_cols].corr()
        
        # Find strongest correlations
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.3:  # Only significant correlations
                    correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                    })
        
        return {
            'matrix': correlation_matrix,
            'significant_correlations': sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
        }

# Utility functions for data quality
def assess_data_quality(df: pd.DataFrame) -> Dict:
    """Assess overall data quality"""
    total_rows = len(df)
    
    quality_report = {
        'total_rows': total_rows,
        'total_columns': len(df.columns),
        'missing_data': {},
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'quality_score': 0.0
    }
    
    # Calculate missing data percentage
    missing_counts = df.isnull().sum()
    quality_report['missing_data'] = {
        col: {
            'count': int(count),
            'percentage': float(count / total_rows * 100)
        }
        for col, count in missing_counts.items() if count > 0
    }
    
    # Calculate overall quality score (0-100)
    missing_percentage = df.isnull().sum().sum() / (total_rows * len(df.columns)) * 100
    duplicate_percentage = quality_report['duplicate_rows'] / total_rows * 100
    
    quality_score = 100 - missing_percentage - duplicate_percentage
    quality_report['quality_score'] = max(0, quality_score)
    
    return quality_report

# Configuration for Streamlit app
STREAMLIT_CONFIG = {
    'page_title': '🔍 Análise de Violência contra a Mulher - SC',
    'page_icon': '📊',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'menu_items': {
        'Get Help': 'https://github.com/your-repo/violence-analysis',
        'Report a bug': 'https://github.com/your-repo/violence-analysis/issues',
        'About': '# Violence Analysis Dashboard\nDesenvolvido para análise de dados de violência contra a mulher em Santa Catarina.'
    }
}

# Export all utilities
__all__ = [
    'COLOR_SCHEMES',
    'SC_MESOREGIONS_DETAILED',
    'CRIME_SEVERITY_MAP',
    'EDUCATION_MAPPING',
    'RELATIONSHIP_MAPPING',
    'DataProcessor',
    'VisualizationHelper',
    'StatisticalAnalyzer',
    'assess_data_quality',
    'STREAMLIT_CONFIG'
]