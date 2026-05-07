import pandas as pd

# Rutas de tus archivos
ruta_entrada = r"C:\Users\JoseFernandoGallegos\OneDrive - Farmacos Darovi\distribuidores multiples.xlsx"
ruta_salida = r"C:\Users\JoseFernandoGallegos\OneDrive - Farmacos Darovi\distribuidores_limpios.xlsx"

# Leer las dos hojas
df = pd.read_excel(ruta_entrada, sheet_name='catalogo quim')
df_hoja2 = pd.read_excel(ruta_entrada, sheet_name='Hoja2')

# Limpiar y unir
df_hoja2 = df_hoja2.dropna(subset=['Laboratorio Distribuidor '])
merged = df.merge(df_hoja2, on='Laboratorio Distribuidor ', how='left')

# Separar y Desdinamizar (Unpivot)
orig_cols = df.columns.tolist()
value_vars = [c for c in df_hoja2.columns if c != 'Laboratorio Distribuidor ']
melted = pd.melt(merged, id_vars=orig_cols, value_vars=value_vars, value_name='Laboratorio_Separado')

# Dar formato final y guardar
exploded = melted.dropna(subset=['Laboratorio_Separado']).copy()
exploded = exploded.drop(columns=['variable', 'Laboratorio Distribuidor '])
exploded = exploded.rename(columns={'Laboratorio_Separado': 'Laboratorio Distribuidor '})
exploded = exploded.sort_values(by=['Código', 'Clave del Artículo']).reset_index(drop=True)

exploded.to_excel(ruta_salida, index=False)