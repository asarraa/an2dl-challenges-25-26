import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from skimage import exposure, feature, metrics, io, color
from skimage.measure import regionprops, label, find_contours
from skimage.filters import gaussian, laplace
from scipy.ndimage import binary_erosion, binary_dilation, center_of_mass
from scipy.stats import entropy
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import umap

# ==========================================
# CONFIGURAZIONE
# ==========================================
DATASET_ROOT = "./training"
MASKS_ROOT = "./masks"
CSV_LABELS = "./train_labels.csv"

CLASSES = ['Luminal A', 'Luminal B', 'HER2+', 'Triple Negative']
np.random.seed(42)

print("="*80)
print("DEEP DATA EXPLORATION - CHALLENGE ISTOPATOLOGIA")
print("Con Analisi Artefatti e Qualità Immagini")
print("="*80)

# ==========================================
# 1. FUNZIONI DETECTION ARTEFATTI
# ==========================================

def detect_artifacts(img_array, sample_index):
    """
    Rileva artefatti comuni in immagini istologiche:
    - Presenze di colori verdi/blu anomali (Shrek, macchie disegnate)
    - Regioni ad alto contrasto con colori specifici
    """
    artifacts = {
        'has_green_marks': False,
        'has_blue_marks': False,
        'has_shrek': False,
        'green_percentage': 0,
        'artifact_confidence': 0
    }
    
    try:
        if len(img_array.shape) != 3 or img_array.shape[2] < 3:
            return artifacts
        
        # Estrai canali
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        
        # Rileva verde anomalo (macchie disegnate)
        # Verde tistalare naturale: G >> R e G >> B, ma con valori ragionevoli
        # Verde disegnato: picco acuto di G
        green_anomaly = (g > 150) & (g > r + 50) & (g > b + 50)
        green_pixels = np.sum(green_anomaly)
        green_percentage = (green_pixels / green_anomaly.size) * 100
        
        if green_percentage > 1:  # Più dell'1% verde anomalo
            artifacts['has_green_marks'] = True
            artifacts['green_percentage'] = green_percentage
        
        # Rileva blu anomalo (disegni)
        blue_anomaly = (b > 150) & (b > r + 50) & (b > g + 30)
        blue_pixels = np.sum(blue_anomaly)
        if (blue_pixels / blue_anomaly.size) * 100 > 1:
            artifacts['has_blue_marks'] = True
        
        # Rileva "Shrek" pattern: presenza massiccia di verde brillante
        shrek_pattern = (g > 200) & ((g - r) > 80) & ((g - b) > 80)
        if np.sum(shrek_pattern) / shrek_pattern.size > 0.05:  # >5% verde brillante
            artifacts['has_shrek'] = True
        
        # Confidence score
        if artifacts['has_shrek']:
            artifacts['artifact_confidence'] = 0.95
        elif artifacts['has_green_marks'] or artifacts['has_blue_marks']:
            artifacts['artifact_confidence'] = 0.7
        
    except Exception as e:
        print(f"Errore detection artefatti: {e}")
    
    return artifacts

def remove_colored_artifacts(img_array, mask_array, color_threshold_g=150):
    """
    Rimuove artefatti colorati (verde, blu) dall'immagine e dalla maschera.
    Sostituisce pixel anomali con media locale.
    """
    try:
        if len(img_array.shape) != 3 or img_array.shape[2] < 3:
            return img_array, mask_array
        
        img_clean = img_array.copy().astype(np.float32)
        r, g, b = img_clean[:, :, 0], img_clean[:, :, 1], img_clean[:, :, 2]
        
        # Maschera per artefatti verdi
        green_artifact = (g > color_threshold_g) & (g > r + 50) & (g > b + 50)
        
        # Maschera per artefatti blu
        blue_artifact = (b > color_threshold_g) & (b > r + 50) & (b > g + 30)
        
        artifact_mask = green_artifact | blue_artifact
        
        # Se ci sono artefatti, sostituisci con media locale
        if np.sum(artifact_mask) > 0:
            # Usa filtro mediano per inpainting semplice
            img_clean[:, :, 0] = cv2.medianBlur(img_clean[:, :, 0].astype(np.uint8), 5).astype(np.float32)
            img_clean[:, :, 1] = cv2.medianBlur(img_clean[:, :, 1].astype(np.uint8), 5).astype(np.float32)
            img_clean[:, :, 2] = cv2.medianBlur(img_clean[:, :, 2].astype(np.uint8), 5).astype(np.float32)
            
            # Azzera anche la maschera negli artefatti
            mask_clean = mask_array.copy()
            mask_clean[artifact_mask] = 0
            
            return img_clean.astype(np.uint8), mask_clean
        
        return img_array, mask_array
    
    except Exception as e:
        print(f"Errore inpainting: {e}")
        return img_array, mask_array

# ==========================================
# 2. CARICAMENTO DATASET DA CSV
# ==========================================

def load_dataset_from_csv(csv_path, img_root, mask_root):
    """Carica dataset dal CSV e associa immagini e maschere."""
    print("\n[1/13] Caricamento Dataset da CSV...")
    
    try:
        df_csv = pd.read_csv(csv_path)
        print(f"✓ CSV caricato: {len(df_csv)} righe")
    except Exception as e:
        print(f"❌ Errore caricamento CSV: {e}")
        return None
    
    if 'label' in df_csv.columns:
        df_csv.rename(columns={'label': 'class'}, inplace=True)
    elif 'Class' in df_csv.columns:
        df_csv.rename(columns={'Class': 'class'}, inplace=True)
    elif 'subtype' in df_csv.columns:
        df_csv.rename(columns={'subtype': 'class'}, inplace=True)
    
    if 'sample_index' not in df_csv.columns:
        print("❌ CSV deve contenere colonna 'sample_index'")
        return None
    
    data_list = []
    for idx, row in df_csv.iterrows():
        sample_index = row['sample_index']
        img_path = os.path.join(img_root, sample_index)
        name_without_ext = os.path.splitext(sample_index)[0]
        #mask_path is mask + filename without img prefix and with .png extension
        mask_name = name_without_ext.replace("img_", "mask_")
        mask_path = os.path.join(mask_root, mask_name + ".png")
        
        if not os.path.exists(img_path):
            continue
        if not os.path.exists(mask_path):
            continue
        
        data_list.append({
            'sample_index': sample_index,
            'class': row['class'],
            'img_path': img_path,
            'mask_path': mask_path
        })
    
    df = pd.DataFrame(data_list)
    print(f"✓ Dataset costruito: {len(df)} immagini valide con maschere")
    return df

df = load_dataset_from_csv(CSV_LABELS, DATASET_ROOT, MASKS_ROOT)
if df is None:
    print("❌ Errore fatale nel caricamento dataset")
    exit(1)

# ==========================================
# 3. ANALISI DIMENSIONI E ARTEFATTI
# ==========================================

def extract_image_mask_properties(img_path, mask_path, clean_artifacts=True):
    """Estrae proprietà e rileva artefatti."""
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        width, height = img.size
        
        mask = Image.open(mask_path).convert('L')
        mask_array = np.array(mask)
        mask_binary = (mask_array > 127).astype(np.uint8)
        
        # Rileva artefatti
        artifacts = detect_artifacts(img_array, os.path.basename(img_path))
        
        # Pulisci artefatti se richiesto
        if clean_artifacts and (artifacts['has_green_marks'] or artifacts['has_blue_marks']):
            img_array, mask_binary = remove_colored_artifacts(img_array, mask_binary)
        
        # Calcola proprietà
        roi_pixels = np.sum(mask_binary)
        total_pixels = mask_binary.size
        roi_coverage = roi_pixels / total_pixels if total_pixels > 0 else 0
        
        centroid = center_of_mass(mask_binary) if roi_pixels > 0 else (height/2, width/2)
        
        labeled_mask = label(mask_binary)
        n_components = len(np.unique(labeled_mask)) - 1
        
        if n_components > 0:
            region_props = regionprops(labeled_mask)
            largest_tumor_area = max([r.area for r in region_props])
            mean_tumor_area = np.mean([r.area for r in region_props])
        else:
            largest_tumor_area = 0
            mean_tumor_area = 0
        
        # Colore
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            color_mean_r = np.mean(img_array[:, :, 0])
            color_mean_g = np.mean(img_array[:, :, 1])
            color_mean_b = np.mean(img_array[:, :, 2])
            color_std_r = np.std(img_array[:, :, 0])
            color_std_g = np.std(img_array[:, :, 1])
            color_std_b = np.std(img_array[:, :, 2])
        else:
            color_mean_r = color_mean_g = color_mean_b = np.mean(img_array)
            color_std_r = color_std_g = color_std_b = np.std(img_array)
        
        # Texture
        img_min = np.min(img_array)
        img_max = np.max(img_array)
        contrast = (img_max - img_min) / (img_max + img_min + 1e-6)
        
        hist, _ = np.histogram(img_array.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        img_entropy = entropy(hist + 1e-10)
        
        gray_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY) if len(img_array.shape)==3 else img_array.astype(np.uint8)
        edges = cv2.Canny(gray_img, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
        
        lbp_variance = np.var(img_array)
        
        return {
            'width': width,
            'height': height,
            'aspect_ratio': width / height if height > 0 else 0,
            'area': width * height,
            'roi_coverage': roi_coverage,
            'tumor_area': roi_pixels,
            'n_components': n_components,
            'largest_tumor_area': largest_tumor_area,
            'mean_tumor_area': mean_tumor_area,
            'centroid_y': centroid[0],
            'centroid_x': centroid[1],
            'color_mean_r': color_mean_r,
            'color_mean_g': color_mean_g,
            'color_mean_b': color_mean_b,
            'color_std_r': color_std_r,
            'color_std_g': color_std_g,
            'color_std_b': color_std_b,
            'contrast': contrast,
            'entropy': img_entropy,
            'edge_density': edge_density,
            'lbp_variance': lbp_variance,
            'has_green_marks': artifacts['has_green_marks'],
            'has_blue_marks': artifacts['has_blue_marks'],
            'has_shrek': artifacts['has_shrek'],
            'green_percentage': artifacts['green_percentage'],
            'artifact_confidence': artifacts['artifact_confidence']
        }
    except Exception as e:
        print(f"❌ Errore: {e}")
        return None

print("\n[2/13] Estrazione proprietà immagini...")
properties_list = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    props = extract_image_mask_properties(row['img_path'], row['mask_path'])
    if props is not None:
        props['sample_index'] = row['sample_index']
        props['class'] = row['class']
        properties_list.append(props)

df_full = pd.DataFrame(properties_list)
print(f"✓ Proprietà estratte: {len(df_full)} immagini")

# ==========================================
# 4. ANALISI ARTEFATTI E QUALITÀ
# ==========================================

print("\n[3/13] Analisi Artefatti...")

n_shrek = df_full['has_shrek'].sum()
n_green = df_full['has_green_marks'].sum()
n_blue = df_full['has_blue_marks'].sum()
n_artifacts = len(df_full[df_full['artifact_confidence'] > 0.5])

print(f"✓ Immagini con problemi:")
print(f"  → Shrek (verdi massicce): {n_shrek}")
print(f"  → Macchie verdi disegnate: {n_green - n_shrek}")
print(f"  → Artefatti blu: {n_blue}")
print(f"  → TOTALE problematiche: {n_artifacts}")

# Visualizza artefatti
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Conteggio artefatti
artifact_data = {
    'Shrek': n_shrek,
    'Macchie Verdi': n_green - n_shrek,
    'Artefatti Blu': n_blue,
    'Pulite': len(df_full) - n_artifacts
}
colors_bar = ['red', 'orange', 'yellow', 'green']
axes[0, 0].bar(artifact_data.keys(), artifact_data.values(), color=colors_bar)
axes[0, 0].set_title('Conteggio Artefatti', fontweight='bold')
axes[0, 0].set_ylabel('# Immagini')
for i, v in enumerate(artifact_data.values()):
    axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')

# Green percentage distribution
df_green = df_full[df_full['has_green_marks']]
axes[0, 1].hist(df_green['green_percentage'], bins=20, color='green', alpha=0.7)
axes[0, 1].set_title('Distribuzione % Verde Anomalo', fontweight='bold')
axes[0, 1].set_xlabel('% Pixel Verdi')
axes[0, 1].set_ylabel('Frequenza')

# Artifact confidence
axes[1, 0].hist(df_full['artifact_confidence'], bins=20, color='orange', alpha=0.7)
axes[1, 0].set_title('Confidence Artefatti', fontweight='bold')
axes[1, 0].set_xlabel('Confidence Score')
axes[1, 0].axvline(0.5, color='red', linestyle='--', label='Threshold')

# Artifact count per classe
artifact_by_class = df_full[df_full['artifact_confidence'] > 0.5].groupby('class').size()
axes[1, 1].bar(artifact_by_class.index, artifact_by_class.values, color='coral')
axes[1, 1].set_title('Artefatti per Classe', fontweight='bold')
axes[1, 1].set_ylabel('# Immagini')

plt.tight_layout()
plt.savefig('00_artifact_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 5. ANALISI DIMENSIONI
# ==========================================

print("\n[4/13] Analisi Dimensioni Immagini...")

# Verifica se una dimensione è sempre 1024
print(f"\n✓ Analisi dimensioni:")
print(f"  → Width: {df_full['width'].min()}-{df_full['width'].max()}")
print(f"  → Height: {df_full['height'].min()}-{df_full['height'].max()}")

n_1024_width = (df_full['width'] == 1024).sum()
n_1024_height = (df_full['height'] == 1024).sum()
print(f"  → Immagini con width=1024: {n_1024_width}")
print(f"  → Immagini con height=1024: {n_1024_height}")

if n_1024_width > len(df_full) * 0.9 or n_1024_height > len(df_full) * 0.9:
    print(f"  ⚠️  UNA DIMENSIONE È FISSA A 1024! Usa Input CNN: 1024xX")
    fixed_dim = 1024
else:
    fixed_dim = None

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

scatter = axes[0, 0].scatter(df_full['width'], df_full['height'], 
                    c=pd.Categorical(df_full['class']).codes, 
                    cmap='tab10', alpha=0.5, s=30)
axes[0, 0].set_xlabel('Larghezza (px)')
axes[0, 0].set_ylabel('Altezza (px)')
axes[0, 0].set_title('Dispersione Dimensioni', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
if fixed_dim:
    axes[0, 0].axvline(1024, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].axhline(1024, color='red', linestyle='--', alpha=0.5)

for cls in df_full['class'].unique():
    data_cls = df_full[df_full['class'] == cls]['aspect_ratio']
    axes[0, 1].hist(data_cls, alpha=0.5, label=cls, bins=20)
axes[0, 1].set_xlabel('Aspect Ratio (W/H)')
axes[0, 1].set_ylabel('Frequenza')
axes[0, 1].set_title('Distribuzione Aspect Ratio', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].axvline(1, color='red', linestyle='--', linewidth=2)

sns.boxplot(x='class', y='width', data=df_full, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Larghezza per Classe', fontweight='bold')
axes[1, 0].set_ylabel('Larghezza (px)')

sns.boxplot(x='class', y='height', data=df_full, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Altezza per Classe', fontweight='bold')
axes[1, 1].set_ylabel('Altezza (px)')

plt.tight_layout()
plt.savefig('01_dimensions_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 6. ANALISI ROI (REGION OF INTEREST)
# ==========================================

print("\n[5/13] Analisi ROI (Region of Interest)...")
print(f"\n✓ ROI Extraction - cosa significa:")
print(f"  → La maschera binaria identifica la REGIONE DI INTERESSE (tumore)")
print(f"  → Anzichè usare l'intera immagine (spesso con sfondo"), 
print(f"  → Usiamo solo l'area marcata dalla maschera (bounding box + cropping)")
print(f"  → Questo riduce rumore e aiuta la CNN a concentrarsi sul tumore")

roi_mean = df_full['roi_coverage'].mean()
print(f"\n✓ ROI Coverage medio: {roi_mean:.2%}")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.boxplot(x='class', y='roi_coverage', data=df_full, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Copertura ROI per Classe', fontweight='bold')
axes[0, 0].set_ylabel('Percentuale ROI')

for cls in df_full['class'].unique():
    data_cls = df_full[df_full['class'] == cls]['roi_coverage']
    axes[0, 1].hist(data_cls, alpha=0.5, label=cls, bins=20)
axes[0, 1].set_xlabel('ROI Coverage')
axes[0, 1].set_ylabel('Frequenza')
axes[0, 1].set_title('Distribuzione Copertura ROI', fontweight='bold')
axes[0, 1].legend()

scatter = axes[1, 0].scatter(df_full['area'], df_full['roi_coverage'] * 100, 
                              c=pd.Categorical(df_full['class']).codes, cmap='tab10', alpha=0.5, s=30)
axes[1, 0].set_xlabel('Area Immagine (px²)')
axes[1, 0].set_ylabel('ROI Coverage (%)')
axes[1, 0].set_title('ROI vs Dimensione Immagine', fontweight='bold')
axes[1, 0].set_xscale('log')
axes[1, 0].grid(True, alpha=0.3)

roi_stats = df_full.groupby('class')['roi_coverage'].agg(['mean', 'std', 'min', 'max'])
axes[1, 1].axis('off')
table_data = [[f"{val:.3f}" for val in roi_stats.loc[cls]] for cls in roi_stats.index]
table = axes[1, 1].table(cellText=table_data, 
                         rowLabels=roi_stats.index,
                         colLabels=['Media', 'Std', 'Min', 'Max'],
                         loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
axes[1, 1].set_title('Statistiche ROI per Classe', fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('02_roi_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 7. ANALISI TEXTURE
# ==========================================

print("\n[6/13] Analisi Texture...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.violinplot(x='class', y='contrast', data=df_full, ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Contrasto per Classe', fontweight='bold')

sns.violinplot(x='class', y='entropy', data=df_full, ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Entropia (Disordine Tessutale)', fontweight='bold')

sns.violinplot(x='class', y='edge_density', data=df_full, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Densità Bordi', fontweight='bold')

sns.violinplot(x='class', y='lbp_variance', data=df_full, ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Varianza LBP (Texture Complexity)', fontweight='bold')

plt.tight_layout()
plt.savefig('03_texture_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

feature_cols = ['contrast', 'entropy', 'edge_density', 'lbp_variance', 'roi_coverage']
corr_matrix = df_full[feature_cols].corr()
top_corr = corr_matrix.abs().unstack().sort_values(ascending=False)[1:6]
print(f"\n✓ Top correlazioni:")
for (f1, f2), corr_val in top_corr.items():
    print(f"  {f1} <-> {f2}: {corr_val:.3f}")

# ==========================================
# 8. RIDUZIONE DIMENSIONALITÀ
# ==========================================

print("\n[7/13] Riduzione Dimensionalità (PCA + UMAP)...")

X_features = df_full[feature_cols].values
X_scaled = StandardScaler().fit_transform(X_features)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"✓ PCA Variance Explained: {pca.explained_variance_ratio_.sum():.2%}")
print("  → Calcolando UMAP...")

reducer = umap.UMAP(n_components=2, min_dist=0.1, n_neighbors=15, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

unique_classes = df_full['class'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_classes)))
for i, cls in enumerate(unique_classes):
    mask = df_full['class'] == cls
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=cls, alpha=0.6, s=30, color=colors[i])
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
axes[0].set_title('PCA: Features 2D', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

for i, cls in enumerate(unique_classes):
    mask = df_full['class'] == cls
    axes[1].scatter(X_umap[mask, 0], X_umap[mask, 1], label=cls, alpha=0.6, s=30, color=colors[i])
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')
axes[1].set_title('UMAP: Features 2D', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('04_dimensionality_reduction.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 9. OUTLIER DETECTION
# ==========================================

print("\n[8/13] Outlier Detection...")

iso_forest = IsolationForest(contamination=0.05, random_state=42)
df_full['outlier'] = iso_forest.fit_predict(X_scaled)
df_full['outlier_score'] = iso_forest.score_samples(X_scaled)

n_outliers = (df_full['outlier'] == -1).sum()
print(f"✓ Outlier rilevati: {n_outliers} ({n_outliers/len(df_full)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for cls in unique_classes:
    mask = (df_full['class'] == cls) & (df_full['outlier'] == 1)
    axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1], label=cls, alpha=0.6, s=30)

outlier_mask = df_full['outlier'] == -1
axes[0].scatter(X_umap[outlier_mask, 0], X_umap[outlier_mask, 1], 
               color='red', marker='X', s=100, label='Outlier', edgecolors='black')
axes[0].set_title('UMAP con Outlier', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(df_full['outlier_score'], bins=30, alpha=0.7, color='skyblue')
axes[1].axvline(df_full[df_full['outlier'] == -1]['outlier_score'].max(), 
               color='red', linestyle='--', label='Threshold')
axes[1].set_xlabel('Outlier Score')
axes[1].set_ylabel('Frequenza')
axes[1].set_title('Distribuzione Outlier Score', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('05_outlier_detection.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Top 5 Outlier/Artefatti:")
outlier_candidates = df_full.nsmallest(5, 'outlier_score')[
    ['sample_index', 'class', 'outlier_score', 'has_shrek', 'has_green_marks']
]
print(outlier_candidates.to_string())

# ==========================================
# 10. STATISTICHE COLORE
# ==========================================

print("\n[9/13] Analisi Statistiche Colore RGB...")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

channels_mean = ['color_mean_r', 'color_mean_g', 'color_mean_b']
colors_list = ['red', 'green', 'blue']
for i, (color, channel) in enumerate(zip(colors_list, channels_mean)):
    sns.boxplot(x='class', y=channel, data=df_full, ax=axes[0, i], palette='Set2')
    axes[0, i].set_title(f'Media Canale {color.upper()}', fontweight='bold')
    axes[0, i].set_ylabel('Valore Medio')

channels_std = ['color_std_r', 'color_std_g', 'color_std_b']
for i, (color, channel) in enumerate(zip(colors_list, channels_std)):
    sns.boxplot(x='class', y=channel, data=df_full, ax=axes[1, i], palette='Set2')
    axes[1, i].set_title(f'Std Dev Canale {color.upper()}', fontweight='bold')
    axes[1, i].set_ylabel('Deviazione Std')

plt.tight_layout()
plt.savefig('06_color_statistics.png', dpi=150, bbox_inches='tight')
plt.show()

mean_r = df_full['color_mean_r'].mean()
mean_g = df_full['color_mean_g'].mean()
mean_b = df_full['color_mean_b'].mean()
std_r = df_full['color_std_r'].mean()
std_g = df_full['color_std_g'].mean()
std_b = df_full['color_std_b'].mean()

print(f"\n✓ PARAMETRI NORMALIZZAZIONE (ImageDataGenerator):")
print(f"  → Mean (0-255): [{mean_r:.1f}, {mean_g:.1f}, {mean_b:.1f}]")
print(f"  → Std (0-255):  [{std_r:.1f}, {std_g:.1f}, {std_b:.1f}]")
print(f"  → Mean (0-1):   [{mean_r/255:.3f}, {mean_g/255:.3f}, {mean_b/255:.3f}]")
print(f"  → Std (0-1):    [{std_r/255:.3f}, {std_g/255:.3f}, {std_b/255:.3f}]")

# ==========================================
# 11. AUGMENTATION STRATEGY
# ==========================================

print("\n[10/13] Strategie Augmentation...")
print(f"\n✓ AUGMENTATION CONSIGLIATA:")
print(f"  → Rotazioni: 0°, 90°, 180°, 270° (tessuto è isotrópico)")
print(f"  → Flip Orizzontale: Sì")
print(f"  → Flip Verticale: Sì")
print(f"  → Traslazioni: Random 10-20% dell'immagine")
print(f"  → Zoom: Random 10-20%")
print(f"  ⚠️  ATTENZIONE: Cambiamenti di luce (Stain Normalization)")
print(f"       → Colore H&E varia tra laboratori")
print(f"       → Usa: albumentations.RandomBrightnessContrast() MODERATO")
print(f"       → O: Colore jittering MAXcauto su canale G (verde)")

# ==========================================
# 12. RACCOMANDAZIONI FINALI
# ==========================================

print("\n[11/13] Analisi Strategie Normalizzazione (Logbook Advice)...")

# Analizza se batch size sarà problematico
df_clean = df_full[df_full['artifact_confidence'] <= 0.5]
dataset_size = len(df_full)
clean_dataset_size = len(df_clean)

print(f"\n✓ DIMENSIONE DATASET E BATCH NORM:")
print(f"  • Dataset totale: {dataset_size} immagini")
print(f"  • Dataset pulito: {clean_dataset_size} immagini")

# Calcola recommended batch sizes
recommended_batch_sizes = []
if clean_dataset_size > 2000:
    recommended_batch_sizes = [32, 64, 128]
elif clean_dataset_size > 1000:
    recommended_batch_sizes = [16, 32, 64]
else:
    recommended_batch_sizes = [8, 16, 32]

print(f"  → Batch sizes consigliati: {recommended_batch_sizes}")

# Logbook advice: piccoli batch = BatchNorm instabile
if clean_dataset_size < 1000:
    norm_strategy = "LayerNorm o InstanceNorm (BATCH PICCOLO)"
    print(f"  ⚠️  PICCOLO DATASET - BatchNorm instabile!")
    print(f"      → Usa LayerNorm o InstanceNorm")
    print(f"      → Oppure: BatchNorm + Dropout maggiore")
else:
    norm_strategy = "BatchNorm (BATCH MEDIO-GRANDE)"
    print(f"  ✓ Dataset abbastanza grande per BatchNorm")
    print(f"      → BatchNorm con batch_size >= {recommended_batch_sizes[0]}")

# Visualizza effetto batch size
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Batch size simulation
batch_sizes = [8, 16, 32, 64, 128]
batch_stabilities = []
for bs in batch_sizes:
    n_batches = clean_dataset_size / bs
    stability = min(1.0, n_batches / 10)  # Più batches = più stabile
    batch_stabilities.append(stability)

axes[0].plot(batch_sizes, batch_stabilities, marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0].fill_between(batch_sizes, batch_stabilities, alpha=0.3, color='steelblue')
axes[0].axhline(0.7, color='orange', linestyle='--', label='Soglia Stabilità')
axes[0].axhline(0.5, color='red', linestyle='--', label='Rischio Instabilità')
axes[0].set_xlabel('Batch Size')
axes[0].set_ylabel('Stabilità BatchNorm Stimata')
axes[0].set_title('Effetto Batch Size su BatchNorm', fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Normalization strategy comparison
norm_types = ['BatchNorm\n(Big Batch)', 'LayerNorm\n(Small Batch)', 'InstanceNorm\n(Per-Sample)', 'GroupNorm\n(Hybrid)']
norm_stability = [0.95 if clean_dataset_size > 1000 else 0.4, 0.85, 0.9, 0.8]
colors_norm = ['green' if s > 0.7 else 'red' for s in norm_stability]

bars = axes[1].bar(norm_types, norm_stability, color=colors_norm, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Stabilità Relativa')
axes[1].set_title('Confronto Strategie Normalizzazione', fontweight='bold')
axes[1].set_ylim([0, 1])
for i, (bar, val) in enumerate(zip(bars, norm_stability)):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.03, f'{val:.2f}', 
                ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('08_normalization_strategy.png', dpi=150, bbox_inches='tight')
plt.show()

# Logbook wisdom interpretation
print(f"\n✓ INTERPRETAZIONE LOGBOOK (04/12):")
print(f"  '...the river flows, but the stone must find its own center...'")
print(f"  → BatchNorm = 'Oceano' (molti dati fluiscono insieme)")
print(f"  → Piccoli dataset = 'Tazza d'acqua' (non usare regole dell'oceano)")
print(f"\n  'Know the dimension of your stream, before the banks you build'")
print(f"  → Il tuo stream (dataset): {clean_dataset_size} immagini")
print(f"  → Scegli normalizzazione in base a QUESTO numero")

print("\n[12/13] Generazione Raccomandazioni...")

class_counts = df_full['class'].value_counts()
class_imbalance = (class_counts.max() - class_counts.min()) / class_counts.mean()

class_weights = {}
for cls in class_counts.index:
    n_samples = class_counts[cls]
    weight = len(df_full) / (len(class_counts) * n_samples)
    class_weights[cls] = weight

recommended_size = int(np.sqrt(df_full['area'].median()))
if fixed_dim:
    recommended_size = fixed_dim

preprocessing_strategy = "Cropping su Maschera (ROI)" if roi_mean < 0.3 else "Resizing standard"

recommendations = {
    'Input Size': f"{recommended_size}x{recommended_size}",
    'Immagini da eliminare': f"Shrek: {n_shrek}, Artefatti: {n_artifacts - n_shrek}",
    'Preprocessing': preprocessing_strategy,
    'ROI Extraction': "Sì - Usare maschera per bounding box",
    'Class Weights': f"Sì - Ratio: {class_imbalance:.2f}" if class_imbalance > 1.5 else "Moderato",
    'Batch Size': f"{recommended_batch_sizes[0]}-{recommended_batch_sizes[-1]} (Dataset: {clean_dataset_size})",
    'Normalizzazione Layer': norm_strategy,
    'Augmentation': "Rotazioni 90°, Flip, Traslazioni moderate",
    'Stain Normalization': "Importante - Luce variabile tra immagini",
    'Normalize Mean (RGB)': f"[{mean_r/255:.3f}, {mean_g/255:.3f}, {mean_b/255:.3f}]",
    'Normalize Std (RGB)': f"[{std_r/255:.3f}, {std_g/255:.3f}, {std_b/255:.3f}]",
    'Outlier Handling': f"Rimuovere {n_outliers} immagini anomale",
}

fig, ax = plt.subplots(figsize=(14, 9))
ax.axis('off')
table_data = [[k, v] for k, v in recommendations.items()]
table = ax.table(cellText=table_data, colLabels=['Parametro', 'Valore Consigliato'],
                 loc='center', cellLoc='left', colWidths=[0.35, 0.65])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
for i in range(len(recommendations) + 1):
    if i == 0:
        table[(i, 0)].set_facecolor('#4CAF50')
        table[(i, 1)].set_facecolor('#4CAF50')
        table[(i, 0)].set_text_props(weight='bold', color='white')
        table[(i, 1)].set_text_props(weight='bold', color='white')
    else:
        if i % 2 == 0:
            table[(i, 0)].set_facecolor('#f0f0f0')
            table[(i, 1)].set_facecolor('#f0f0f0')

plt.title('RACCOMANDAZIONI FINALI PER CNN', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('07_recommendations.png', dpi=150, bbox_inches='tight')
plt.show()

# ==========================================
# 13. EXPORT E SUMMARY
# ==========================================

print("\n[12/13] Export Dataset...")

# Salva problematic images
df_artifacts = df_full[df_full['artifact_confidence'] > 0.5][
    ['sample_index', 'class', 'has_shrek', 'has_green_marks', 'green_percentage', 'artifact_confidence']
].sort_values('artifact_confidence', ascending=False)
df_artifacts.to_csv('images_to_review.csv', index=False)
print(f"✓ Immagini da revisionare: images_to_review.csv ({len(df_artifacts)} immagini)")

# Salva outliers
df_outliers = df_full[df_full['outlier'] == -1][
    ['sample_index', 'class', 'outlier_score', 'contrast', 'roi_coverage']
].sort_values('outlier_score')
df_outliers.to_csv('outliers_to_check.csv', index=False)
print(f"✓ Outliers: outliers_to_check.csv ({len(df_outliers)} immagini)")

# Salva clean images
print(f"✓ Immagini pulite: {len(df_clean)} ({len(df_clean)/len(df_full)*100:.1f}%)")

# Salva dataset completo
df_full.to_csv('dataset_with_features.csv', index=False)
print(f"✓ Dataset completo: dataset_with_features.csv")

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "="*80)
print("RIEPILOGO FINALE")
print("="*80)
print(f"\n📊 DATASET STATS:")
print(f"  • Totale immagini: {len(df_full)}")
print(f"  • Immagini pulite: {len(df_clean)} ({len(df_clean)/len(df_full)*100:.1f}%)")
print(f"  • Immagini problematiche: {len(df_artifacts)} ({len(df_artifacts)/len(df_full)*100:.1f}%)")
print(f"  • Outliers rilevati: {len(df_outliers)} ({len(df_outliers)/len(df_full)*100:.1f}%)")

print(f"\n🎨 ARTEFATTI:")
print(f"  • Shrek (verde massiccia): {n_shrek}")
print(f"  • Macchie verdi disegnate: {n_green - n_shrek}")
print(f"  • Artefatti blu: {n_blue}")

print(f"\n📐 DIMENSIONI:")
print(f"  • Media: {df_full['width'].mean():.0f}x{df_full['height'].mean():.0f}")
print(f"  • Input CNN consigliato: {recommended_size}x{recommended_size}")
if fixed_dim:
    print(f"  ⚠️  Dimensione fissa rilevata: {fixed_dim}")

print(f"\n📈 ROI:")
print(f"  • Coverage medio: {roi_mean:.2%}")
print(f"  • Strategia: {preprocessing_strategy}")

print(f"\n⚖️  CLASS IMBALANCE:")
for cls, count in class_counts.items():
    pct = (count / len(df_full)) * 100
    weight = class_weights[cls]
    print(f"  • {cls}: {count} ({pct:.1f}%) - Weight: {weight:.3f}")

print(f"\n🖼️  AUGMENTATION:")
print(f"  • Rotazioni: 0°, 90°, 180°, 270°")
print(f"  • Flip: Orizzontale + Verticale")
print(f"  • Traslazioni: 10-20%")
print(f"  • Zoom: 10-20%")
print(f"  • Stain Norm: IMPORTANTE (verde varia tra lab)")

print("\n✓ GRAFICI SALVATI:")
graphs = [
    '00_artifact_analysis.png',
    '01_dimensions_analysis.png',
    '02_roi_analysis.png',
    '03_texture_analysis.png',
    '04_dimensionality_reduction.png',
    '05_outlier_detection.png',
    '06_color_statistics.png',
    '07_recommendations.png',
]
for graph in graphs:
    print(f"  - {graph}")

print(f"\n✓ CSV SALVATI:")
print(f"  - dataset_with_features.csv (COMPLETO)")
print(f"  - images_to_review.csv ({len(df_artifacts)} artefatti)")
print(f"  - outliers_to_check.csv ({len(df_outliers)} outliers)")

print("\n" + "="*80)
print("PRONTO PER I LABORATORI DI CNN E TRANSFER LEARNING!")
print("="*80)