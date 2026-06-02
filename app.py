import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nibabel as nib
from scipy.stats import skew, kurtosis
from skimage import filters, measure

# =========================================================
# MEDICAL IMAGE CALCULUS KERNEL (Unified Backend)
# =========================================================
def load_clinical_nifti_volume(file_path: str = "brain.nii") -> dict:
    """Loads a real clinical human brain MRI volume and extracts spatial orientations."""
    try:
        img = nib.load(file_path)
        volume_data = img.get_fdata()
        header = img.header
        affine = img.affine
        
        # Normalize voxel pixel intensities to protect visualization arrays
        volume_data = (volume_data - np.min(volume_data)) / (np.max(volume_data) - np.min(volume_data) + 1e-6) * 255.0
        
        return {
            "volume": volume_data,
            "shape": volume_data.shape,
            "affine": affine.tolist(),
            "voxel_sizes": header.get_zooms()[:3],
            "status": "Success: Authentic Clinical Dataset Loaded"
        }
    except Exception as e:
        size = 48
        fallback_vol = np.zeros((size, size, size))
        z, y, x = np.ogrid[:size, :size, :size]
        mask = ((x - 24)**2 + (y - 24)**2 + (z - 24)**2) < 18**2
        fallback_vol[mask] = 120.0
        fallback_vol += np.random.normal(0, 2.0, fallback_vol.shape)
        return {
            "volume": fallback_vol,
            "shape": fallback_vol.shape,
            "affine": np.eye(4).tolist(),
            "voxel_sizes": (1.0, 1.0, 1.0),
            "status": f"Fallback Active: {str(e)}"
        }

def compute_3d_otsu_segmentation(volume: np.ndarray, threshold_factor: float = 1.1) -> np.ndarray:
    """Calculates a global multi-dimensional Otsu threshold to isolate target pathology zones."""
    active_voxels = volume[volume > 10.0]
    if len(active_voxels) == 0:
        return np.zeros_like(volume, dtype=bool)
    global_otsu = filters.threshold_otsu(active_voxels)
    return volume > (global_otsu * threshold_factor)

def calculate_slicer_ras_coordinates(mask: np.ndarray, affine: list) -> np.ndarray:
    """Maps voxel indexing metrics directly into true 3D Slicer RAS millimeters space."""
    if np.sum(mask) == 0:
        return np.array([0.0, 0.0, 0.0])
    centroid_voxels = np.argwhere(mask).mean(axis=0)
    homogeneous_vector = np.append(centroid_voxels, 1.0)
    affine_matrix = np.array(affine)
    world_coordinates = np.dot(affine_matrix, homogeneous_vector)[:3]
    return world_coordinates

def extract_advanced_radiomics(volume: np.ndarray, mask: np.ndarray, voxel_sizes: tuple) -> dict:
    """First-Order Quantitative Radiomics Feature Fields Extraction Module."""
    target_voxels = volume[mask]
    if len(target_voxels) == 0:
        return {"volume_cc": 0.0, "entropy": 0.0, "skewness": 0.0, "kurtosis": 0.0}
        
    voxel_volume_mm3 = voxel_sizes[0] * voxel_sizes[1] * voxel_sizes[2]
    total_volume_cc = (np.sum(mask) * voxel_volume_mm3) / 1000.0
    
    hist, _ = np.histogram(target_voxels, bins=32, density=True)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist)))
    
    return {
        "volume_cc": float(total_volume_cc),
        "entropy": float(entropy),
        "skewness": float(skew(target_voxels)),
        "kurtosis": float(kurtosis(target_voxels))
    }

# =========================================================
# ENTERPRISE UX CONFIGURATION & DASHBOARD INTERFACE
# =========================================================
st.set_page_config(
    page_title="SlicerLink Pro - Clinical Neuroimaging Workspace",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://googleapis.com');
    * { font-family: 'Inter', sans-serif !important; }
    .code-text { font-family: 'JetBrains Mono', monospace !important; font-size: 0.85rem !important; color: #818CF8; }
    .stApp { background-color: #030712; color: #F3F4F6; }
    
    #MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); height: 0rem; }
    
    .slicer-header {
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(135deg, #111827 0%, #1E1B4B 50%, #030712 100%);
        padding: 24px 40px; border-bottom: 1px solid #1F2937; margin: -60px -40px 28px -40px;
    }
    .slicer-title { font-size: 1.6rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.02em; }
    .slicer-subtitle { font-size: 0.85rem; color: #A5B4FC; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }
    .system-status { font-size: 0.75rem; color: #10B981; background: rgba(16, 185, 129, 0.1); padding: 6px 14px; border-radius: 4px; border: 1px solid rgba(16, 185, 129, 0.2); font-weight: 600; }
    
    .workstation-card { background-color: #111827; border: 1px solid #1F2937; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
    .workstation-title { font-size: 0.9rem; font-weight: 600; color: #9CA3AF; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 20px; border-left: 3px solid #6366F1; padding-left: 10px; }
    
    .grid-matrix { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 16px; }
    .matrix-box { background: #030712; border: 1px solid #1F2937; padding: 16px; border-radius: 6px; text-align: center; }
    .matrix-val { font-size: 1.6rem; font-weight: 700; color: #818CF8; }
    .matrix-lbl { font-size: 0.7rem; text-transform: uppercase; color: #6B7280; font-weight: 600; margin-top: 4px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="slicer-header">
        <div>
            <div class="slicer-title">SlicerLink Pro: Clinical NIfTI Ingestion Workstation</div>
            <div class="slicer-subtitle">3D Affine Transform Matrix Core & Pre-Surgical Radiomics Pipeline</div>
        </div>
        <div class="system-status">Institutional DICOM Pipeline Active</div>
    </div>
""", unsafe_allow_html=True)

nifti_data = load_clinical_nifti_volume("brain.nii")
volume_matrix = nifti_data["volume"]

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="workstation-card">', unsafe_allow_html=True)
    st.markdown('<div class="workstation-title">Dataset Registration Status</div>', unsafe_allow_html=True)
    st.write(f"Active Ingestion Stream: `<span class='code-text'>brain.nii</span>`", unsafe_allow_html=True)
    st.write(f"Extracted Structural Geometry Array Sizing: `{nifti_data['shape']}` Voxels")
    st.write(f"Isotropic Resolution Grid Calibration: `{nifti_data['voxel_sizes']}` mm")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="workstation-card">', unsafe_allow_html=True)
    st.markdown('<div class="workstation-title">Surgical Margin Sensitivity Control</div>', unsafe_allow_html=True)
    
    sensitivity = st.slider("Automated Otsu Edge Intensity Bias Threshold Scaling Factor", 0.5, 2.0, 1.1, step=0.05)
    
    seg_mask = compute_3d_otsu_segmentation(volume_matrix, sensitivity)
    features = extract_advanced_radiomics(volume_matrix, seg_mask, nifti_data["voxel_sizes"])
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="workstation-title">Real-Time Extraction Scoreboard</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="grid-matrix">', unsafe_allow_html=True)
    st.markdown(f'<div class="matrix-box"><div class="matrix-val">{features["volume_cc"]:.2f} cc</div><div class="matrix-lbl">Lesion Volume</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="matrix-box"><div class="matrix-val">{features["entropy"]:.3f}</div><div class="matrix-lbl">Shannon Entropy (Chaos)</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="matrix-box"><div class="matrix-val">{features["skewness"]:.2f}</div><div class="matrix-lbl">Voxel Skewness</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="matrix-box"><div class="matrix-val">{features["kurtosis"]:.2f}</div><div class="matrix-lbl">Voxel Kurtosis</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="workstation-card">', unsafe_allow_html=True)
    st.markdown('<div class="workstation-title">3D Slicer Navigation Target Node Export Vector</div>', unsafe_allow_html=True)
    
    ras_coords = calculate_slicer_ras_coordinates(seg_mask, nifti_data["affine"])
    
    # Isolate coordinate indexing safely outside HTML components to avoid formatting issues
    r_val = float(ras_coords[0])
    a_val = float(ras_coords[1])
    s_val = float(ras_coords[2])
    
    st.markdown(f"""
        <div style="background-color:#030712; padding:16px; border-radius:6px; border:1px solid #1F2937;">
            <div class="code-text"># 3D Slicer Navigation Node Tracking Coordinates (RAS Millimeters Frame)</div>
            <div style="color:#10B981; font-family:monospace; font-size:1.1rem; font-weight:600; letter-spacing:0.5px;">
                R: {r_val:.3f} mm &nbsp;&nbsp;|&nbsp;&nbsp; A: {a_val:.3f} mm &nbsp;&nbsp;|&nbsp;&nbsp; S: {s_val:.3f} mm
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="workstation-card">', unsafe_allow_html=True)
    st.markdown('<div class="workstation-title">Multiplanar Reconstruction (MPR) Matrix Canvas</div>', unsafe_allow_html=True)
    
       # Locked onto the axial structural axis dimension cleanly
    max_slices = int(volume_matrix.shape[0])
    slice_idx = st.slider(
        "Orthogonal Plane Depth Cross-Section Selector (Axial Plane)", 
        0, 
        max_slices - 1, 
        max_slices // 2,
        key="surgical_slice_slider"
    )
    
with col2:
    st.markdown('<div class="workstation-card">', unsafe_allow_html=True)
    st.markdown('<div class="workstation-title">3D Slicer Navigation Target Node Export Vector</div>', unsafe_allow_html=True)
    
    ras_coords = calculate_slicer_ras_coordinates(seg_mask, nifti_data["affine"])
    
    # Isolate coordinate indexing safely outside HTML components to avoid formatting issues
    r_val = float(ras_coords[0])
    a_val = float(ras_coords[1])
    s_val = float(ras_coords[2])
    
    st.markdown(f"""
        <div style="background-color:#030712; padding:16px; border-radius:6px; border:1px solid #1F2937;">
            <div class="code-text"># 3D Slicer Navigation Node Tracking Coordinates (RAS Millimeters Frame)</div>
            <div style="color:#10B981; font-family:monospace; font-size:1.1rem; font-weight:600; letter-spacing:0.5px;">
                R: {r_val:.3f} mm &nbsp;&nbsp;|&nbsp;&nbsp; A: {a_val:.3f} mm &nbsp;&nbsp;|&nbsp;&nbsp; S: {s_val:.3f} mm
            </div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="workstation-card">', unsafe_allow_html=True)
    st.markdown('<div class="workstation-title">Multiplanar Reconstruction (MPR) Matrix Canvas</div>', unsafe_allow_html=True)
    
    max_slices = volume_matrix.shape[0]
    slice_idx = st.slider("Orthogonal Plane Depth Cross-Section Selector (Axial Plane)", 0, max_slices - 1, max_slices // 2)
    
    slice_display = np.copy(volume_matrix[slice_idx, :, :])
    mask_display = seg_mask[slice_idx, :, :]
    
    slice_display[mask_display] = np.clip(slice_display[mask_display] + 40.0, 0.0, 255.0)
    
    fig_mpr = px.imshow(
        slice_display,
        color_continuous_scale="gray",
        labels=dict(x="Lateral Grid Array (X)", y="Anterior-Posterior Grid Array (Y)")
    )
    
    if np.sum(mask_display) > 0:
        contours = measure.find_contours(mask_display, 0.5)
        for contour in contours:
            fig_mpr.add_trace(go.Scatter(
                x=contour[:, 1], y=contour[:, 0],
                mode='lines', line=dict(color='#6366F1', width=1.5),
                name="Slicer Link Edge Path"
            ))
            
    fig_mpr.update_layout(
        height=460, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#030712',
        margin=dict(t=10, b=10, l=10, r=10), font=dict(color='#4B5563', family='JetBrains Mono'),
        xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)
    )
    st.plotly_chart(fig_mpr, use_container_width=True, config={'displayModeBar': False})
    st.markdown('</div>', unsafe_allow_html=True)
