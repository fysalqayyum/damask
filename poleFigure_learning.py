import dash
from dash import dcc, html, Input, Output, State, no_update, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from scipy.spatial.transform import Rotation as R
from itertools import permutations, product # Import earlier
import warnings

# ==============================================================================
# 1. Basic Constants (Independent)
# ==============================================================================
pio.templates.default = "plotly_white" # Use a clean template
HCP_CA_RATIO = 1.624 # Default c/a ratio for HCP (e.g., Mg)
DEFAULT_CRYSTAL_SIZE = 0.8
GAP_FRACTION = 0.1

# ==============================================================================
# 2. Helper Functions (Geometry & Crystallography)
# ==============================================================================

# region Geometry Generation
def get_cube_mesh(center=(0, 0, 0), size=1.0, color='blue', opacity=1.0, name='cube'):
    s = size / 2.0; x, y, z = center
    v = np.array([[x-s,y-s,z-s],[x+s,y-s,z-s],[x+s,y+s,z-s],[x-s,y+s,z-s],[x-s,y-s,z+s],[x+s,y-s,z+s],[x+s,y+s,z+s],[x-s,y+s,z+s]])
    f = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]])
    return go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=color,opacity=opacity,flatshading=True,name=name,hoverinfo='name')

def get_hex_prism_mesh(center=(0, 0, 0), size=1.0, height_factor=1.633, color='green', opacity=1.0, name='hex'):
    s = size / 2.0; h = s * height_factor / 2.0; x, y, z = center; v = []; f_new = []; f_top = []; f_sides = []
    for i in range(6): angle = np.pi/3*i; v.append([x+s*np.cos(angle),y+s*np.sin(angle),z-h])
    for i in range(6): angle = np.pi/3*i; v.append([x+s*np.cos(angle),y+s*np.sin(angle),z+h])
    v = np.array(v)
    for i in range(1, 5): f_new.append([0, i, i+1]) # Bottom triangles
    f_top = [[6, 7, 8], [6, 8, 9], [6, 9, 10], [6, 10, 11]] # Correct Top Face Triangulation
    for i in range(6): i_next=(i+1)%6; f_sides.extend([[i,i_next,i_next+6],[i,i_next+6,i+6]])
    f = np.array(f_new + f_top + f_sides) # Combine bottom, corrected top, sides
    return go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=color,opacity=opacity,flatshading=True,name=name,hoverinfo='name')

def get_plane_mesh(normal, center, size=1.0, color='rgba(255, 0, 0, 0.5)', name='plane'):
    # Ensure normal is a numpy array and normalized
    normal = np.asarray(normal)
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-8: return go.Mesh3d() # Return empty mesh if normal is zero
    normal = normal / norm_val

    v1 = np.array([1.,0.,0.]) if np.abs(normal[0])<0.9 else np.array([0.,1.,0.])
    v1 -= v1.dot(normal) * normal
    v1_norm = np.linalg.norm(v1)
    if v1_norm < 1e-8: # Handle case where normal is aligned with initial v1 guess
         v1 = np.array([0., 0., 1.])
         v1 -= v1.dot(normal) * normal
         v1_norm = np.linalg.norm(v1)
         if v1_norm < 1e-8: # If normal is also Z, something's wrong, but try Y
             v1 = np.array([0., 1., 0.])
             v1 -= v1.dot(normal) * normal
             v1_norm = np.linalg.norm(v1)

    v1 /= v1_norm
    v2 = np.cross(normal, v1)

    s=size/2.; p1=center+s*v1+s*v2; p2=center-s*v1+s*v2; p3=center-s*v1-s*v2; p4=center+s*v1-s*v2
    v=np.array([p1,p2,p3,p4]); f=np.array([[0,1,2],[0,2,3]])
    return go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],i=f[:,0],j=f[:,1],k=f[:,2],color=color,opacity=0.5,flatshading=True,name=name,hoverinfo='name')
# endregion Geometry Generation

def create_improved_axes(axis_len):
    """Create improved axes with clearer labels for the crystal visualization"""
    x_line = go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines', 
                         line=dict(color='red', width=5), name='X-axis', hoverinfo='name')
    y_line = go.Scatter3d(x=[0, 0], y=[0, axis_len], z=[0, 0], mode='lines', 
                         line=dict(color='green', width=5), name='Y-axis', hoverinfo='name')
    z_line = go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_len], mode='lines', 
                         line=dict(color='blue', width=5), name='Z-axis', hoverinfo='name')
    
    # Add larger, more visible labels with consistent offsets
    label_offset = axis_len * 0.1
    axes_labels = go.Scatter3d(
        x=[axis_len + label_offset, 0, 0],
        y=[0, axis_len + label_offset, 0],
        z=[0, 0, axis_len + label_offset], 
        mode='text', 
        text=['X', 'Y', 'Z'], 
        textfont=dict(color=['red', 'green', 'blue'], size=18, family='Arial Black'),
        hoverinfo='none'
    )
    
    return [x_line, y_line, z_line, axes_labels]

# region Crystallography Helpers
def get_hcp_symmetry_ops():
    ops = [R.identity()]
    base_rots_z = [R.from_rotvec([0, 0, np.deg2rad(60 * i)]) for i in range(1, 6)]
    rot180_x = R.from_rotvec([np.pi, 0, 0])
    final_ops = list(base_rots_z) + [R.identity()]
    for rot_z in final_ops[:]: final_ops.append(rot180_x * rot_z)
    unique_quats = np.unique(np.round([r.as_quat() for r in final_ops], decimals=5), axis=0)
    if len(unique_quats) != 12: warnings.warn(f"Expected 12 HCP rotational ops, found {len(unique_quats)}.")
    return R.from_quat(unique_quats)

HCP_SYMMETRY_OPS = get_hcp_symmetry_ops() # Calculate once

def hkil_to_cartesian_normal(h, k, i, l, c_a_ratio):
    if not np.isclose(i, -(h+k)): warnings.warn(f"Index i!=-(h+k) for ({h}{k}{i}{l}). Using h,k."); i = -(h+k)
    nx = h; ny = (2*k + h) / np.sqrt(3); nz = l / c_a_ratio
    vec = np.array([nx, ny, nz]); norm = np.linalg.norm(vec)
    if norm < 1e-8: return np.array([0., 0., 0.])
    return vec / norm

def get_hcp_normals(h, k, l, c_a_ratio):
    i = -(h+k); initial_normal = hkil_to_cartesian_normal(h, k, i, l, c_a_ratio)
    if np.linalg.norm(initial_normal) < 1e-6: return np.empty((0,3))
    all_normals = HCP_SYMMETRY_OPS.apply(initial_normal)
    additional_normals = []
    if l != 0:
        neg_l_normal = hkil_to_cartesian_normal(h, k, i, -l, c_a_ratio)
        additional_normals = HCP_SYMMETRY_OPS.apply(neg_l_normal)
        combined_normals = np.vstack((all_normals, additional_normals))
    else: combined_normals = all_normals
    norms = np.linalg.norm(combined_normals, axis=1, keepdims=True); valid_norms = norms > 1e-8
    normalized_normals = np.divide(combined_normals, norms, where=valid_norms, out=np.zeros_like(combined_normals))
    valid_normals = normalized_normals[np.linalg.norm(normalized_normals, axis=1) > 1e-6]
    if valid_normals.shape[0] == 0: return np.empty((0,3))
    unique_normals_array = np.unique(np.round(valid_normals, decimals=5), axis=0)
    return unique_normals_array

def get_plane_family_normals(plane_family_str, crystal_type, c_a_ratio=HCP_CA_RATIO):
    hkl_str = plane_family_str.strip('{}')
    if crystal_type == "HCP":
        # Use a mapping for common HCP plane notations
        hcp_plane_mapping = {
            "0001": (0, 0, 0, 1),
            "10-10": (1, 0, -1, 0),
            "10-11": (1, 0, -1, 1),
            "11-20": (1, 1, -2, 0),
            "11-22": (1, 1, -2, 2)
        }
        
        try:
            if hkl_str in hcp_plane_mapping:
                h, k, i, l = hcp_plane_mapping[hkl_str]
                return get_hcp_normals(h, k, l, c_a_ratio)
            else:
                # For more complex cases, we could implement a general parser
                raise ValueError(f"Unrecognized HCP plane: {hkl_str}")
                
        except Exception as e: 
            warnings.warn(f"Cannot parse HCP plane {plane_family_str}: {e}")
            return np.empty((0,3))
    elif crystal_type == "CUBE":
        try:
             indices_str = list(hkl_str); indices = []; i = 0
             while i < len(indices_str):
                 sign = 1
                 if indices_str[i] == '-': sign = -1; i += 1
                 if i < len(indices_str) and indices_str[i].isdigit():
                     indices.append(sign * int(indices_str[i])); i += 1
                 else: raise ValueError(f"Invalid format near index {i}")
             if len(indices) != 3: raise ValueError("Cubic requires 3 indices")
        except Exception as e: warnings.warn(f"Cannot parse cubic plane {plane_family_str}: {e}"); return np.empty((0,3))
        unique_normals = set(); abs_indices = np.abs(indices)
        signs = list(product([1, -1], repeat=3))
        for perm in set(permutations(abs_indices)):
            p_arr = np.array(perm)
            for sgn in signs:
                vec = p_arr * sgn
                if np.all(vec == 0): continue
                norm = np.linalg.norm(vec)
                if norm > 1e-6: unique_normals.add(tuple(vec / norm))
        if not unique_normals: return np.empty((0,3))
        return np.array(list(unique_normals))
    else: warnings.warn(f"Unknown crystal type: {crystal_type}"); return np.empty((0,3))
    
def stereographic_projection(vectors, projection_pole_axis='z'):
    vectors = np.asarray(vectors);
    if vectors.ndim == 1: vectors = vectors[np.newaxis, :]
    if vectors.shape[0] == 0 or vectors.shape[1] != 3 : return np.empty((0, 2))
    rot = R.identity()
    if projection_pole_axis == 'x': rot = R.from_euler('y', 90, degrees=True)
    elif projection_pole_axis == 'y': rot = R.from_euler('x', -90, degrees=True)
    vectors = rot.apply(vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True); valid_norms = norms > 1e-8
    vectors = np.divide(vectors, norms, where=valid_norms, out=np.zeros_like(vectors))
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    upper_hemisphere_mask = z >= -1e-9
    x_upper, y_upper, z_upper = x[upper_hemisphere_mask], y[upper_hemisphere_mask], z[upper_hemisphere_mask]
    if len(z_upper) == 0: return np.empty((0, 2))
    denom = 1 + z_upper
    proj_x = np.divide(x_upper, denom, where=np.abs(denom) > 1e-8, out=np.zeros_like(x_upper))
    proj_y = np.divide(y_upper, denom, where=np.abs(denom) > 1e-8, out=np.zeros_like(y_upper))
    radius_sq = proj_x**2 + proj_y**2; inside_mask = radius_sq <= 1.0001
    return np.stack([proj_x[inside_mask], proj_y[inside_mask]], axis=-1)

def generate_orientations(n_crystals, distribution='Default (Aligned)', diffusivity=0.0, crystal_type="CUBE"):
    if n_crystals <= 0: return R.identity()
    if distribution == 'Random': return R.random(n_crystals)
    target_rot = R.identity(); dist_params = TEXTURE_COMPONENTS.get(distribution)
    if dist_params:
        angles, required_type_pattern = dist_params
        if crystal_type == required_type_pattern: target_rot = R.from_euler('ZXZ', angles, degrees=True)
        else: warnings.warn(f"Texture '{distribution}' invalid for '{crystal_type}'. Using default.")
    if n_crystals == 1: orientations = target_rot
    else: orientations = R.from_quat(np.tile(target_rot.as_quat(), (n_crystals, 1)))
    if diffusivity > 0:
        max_angle = np.pi / 2 * diffusivity
        perturb_magnitudes = np.random.uniform(0, max_angle, n_crystals)
        perturb_axes = R.random(n_crystals).as_rotvec()
        norms = np.linalg.norm(perturb_axes, axis=1, keepdims=True); valid_norms = norms > 1e-8
        perturb_axes = np.divide(perturb_axes, norms, where=valid_norms, out=np.zeros_like(perturb_axes))
        perturb_rots = R.from_rotvec(perturb_magnitudes[:, np.newaxis] * perturb_axes)
        if n_crystals == 1: orientations = perturb_rots * orientations
        else:
             if orientations.single: orientations = R.from_quat(np.tile(orientations.as_quat(), (n_crystals, 1)))
             orientations = perturb_rots * orientations
    return orientations
# endregion Crystallography Helpers

# ==============================================================================
# 3. Dependent Constants (Need Helpers Defined Above)
# ==============================================================================
CUBE_SLIP_SYSTEMS = {
    "None": {"planes": []}, # Use empty list for None case
    "{111}": {"planes": get_plane_family_normals("{111}", "CUBE")},
    "{110}": {"planes": get_plane_family_normals("{110}", "CUBE")},
    "{112}": {"planes": get_plane_family_normals("{112}", "CUBE")},
}
# Define HCP slip systems
HCP_SLIP_SYSTEMS = {
    "None": {"planes": []},
    "Basal {0001}": {"planes": get_plane_family_normals("{0001}", "HCP")},
    "Prismatic {10-10}": {"planes": get_plane_family_normals("{10-10}", "HCP")},
    "Pyramidal {10-11}": {"planes": get_plane_family_normals("{10-11}", "HCP")},
    "2nd Order Pyramidal {11-22}": {"planes": get_plane_family_normals("{11-22}", "HCP")},
}

CRYSTAL_STRUCTURES = {
    "CUBE": {
        "shape": "cube", "slip_systems": CUBE_SLIP_SYSTEMS,
        "pole_families": ["{100}", "{110}", "{111}", "{112}"]
    },
    "HCP": {
        "shape": "hex", "c_a": HCP_CA_RATIO, "slip_systems": HCP_SLIP_SYSTEMS,
        "pole_families": ["{0001}", "{10-10}", "{10-11}", "{11-20}", "{11-22}"]
    },
}

CRYSTAL_STRUCTURES = {
    "CUBE": {
        "shape": "cube", "slip_systems": CUBE_SLIP_SYSTEMS,
        "pole_families": ["{100}", "{110}", "{111}", "{112}"]
    },
    "HCP": {
        "shape": "hex", "c_a": HCP_CA_RATIO, "slip_systems": HCP_SLIP_SYSTEMS,
        "pole_families": ["{0001}", "{10-10}", "{10-11}", "{11-20}", "{11-22}"]
    },
}

TEXTURE_COMPONENTS = {
    "Cube": ([0, 0, 0], "CUBE"), "Goss": ([0, 45, 0], "CUBE"), "Brass": ([35, 45, 0], "CUBE"),
    "Copper": ([90, 35, 45], "CUBE"), "S": ([59, 37, 63], "CUBE"),
    "Basal": ([0, 0, 0], "HCP"), "Prismatic": ([0, 90, 0], "HCP"),
    "RD || <11-20>": ([90, 90, 0], "HCP"),
}
ORIENTATION_DISTRIBUTIONS = ['Default (Aligned)', 'Random'] + list(TEXTURE_COMPONENTS.keys())

DEFAULT_GRID_SIZE = 1
DEFAULT_STRUCTURE = "CUBE"

# ==============================================================================
# 4. Dash App Setup
# ==============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN], suppress_callback_exceptions=True)
server = app.server

# ==============================================================================
# 5. UI Layout Definition
# ==============================================================================
# region Control Panel Components
controls_card = dbc.Card([
    dbc.CardHeader("Configuration"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                dbc.Label("Grid Size:", html_for='grid-size-dropdown', className="small mb-1"),
                dcc.Dropdown(id='grid-size-dropdown', options=[{'label': ("1x1" if s==1 else f"{s}x{s}"), 'value': s} for s in [1, 2, 4, 8, 10, 16]], value=DEFAULT_GRID_SIZE, clearable=False, className="small")
            ], width=6),
            dbc.Col([
                 dbc.Label("Structure:", html_for='crystal-type-dropdown', className="small mb-1"),
                 dcc.Dropdown(id='crystal-type-dropdown', options=[{'label': k, 'value': k} for k in CRYSTAL_STRUCTURES.keys()], value=DEFAULT_STRUCTURE, clearable=False, className="small")
            ], width=6),
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                 dbc.Label("Orientation Texture:", html_for='orientation-dist-dropdown', className="small mb-1"),
                 dbc.Tooltip("Set the initial texture component or distribution.", target='orientation-dist-dropdown'),
                 dcc.Dropdown(id='orientation-dist-dropdown', options=[{'label': d, 'value': d} for d in ORIENTATION_DISTRIBUTIONS], value='Default (Aligned)', clearable=False, className="small"),
            ])
        ], className="mb-2"),
        dbc.Row([
            dbc.Col([
                 dbc.Label("Diffusivity:", html_for='diffusivity-slider', className="small mb-1"),
                 dbc.Tooltip("Controls the angular spread of orientations (0=sharp, 1=wide).", target='diffusivity-slider'),
                 dcc.Slider(id='diffusivity-slider', min=0, max=1, step=0.05, value=0, marks={i/10: f"{i/10:.1f}" for i in range(0, 11, 2)}),
            ])
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dbc.Button("Reset", id='reset-button', color='warning', size="sm"), width="auto"),
            dbc.Col(dbc.Button("Randomize", id='random-button', color='info', size="sm"), width="auto"),
        ], justify="start", className="mb-3"),
        html.Hr(style={'marginTop': 0, 'marginBottom': 10}),
        dbc.Label("Crystal Display", style={'fontWeight':'bold'}, className="small mb-1"),
        dbc.Row([
            dbc.Col([
                dbc.Label("Show Slip Planes:", html_for='slip-system-dropdown', className="small mb-1"),
                dcc.Dropdown(id='slip-system-dropdown', options=[{'label':'None', 'value':'None'}], value='None', clearable=False, className="small"),
            ], width=12)
        ], className="mb-2"),
        html.Div([
            dbc.Label("Selected Crystal Controls", style={'fontWeight':'bold'}, className="small mb-1"),
            dbc.Row([
                dbc.Col(dbc.Label("Index:", className="small"), width='auto'),
                dbc.Col(dcc.Input(id='selected-crystal-index', type='number', value=-1, disabled=True, style={'width': '60px', 'height':'30px'}, className="small"), width='auto'),
            ], align="center", className="mb-1"),
            html.Div(id='euler-sliders-div', style={'display': 'none'}, children=[
                dbc.Label("Euler Angles (Bunge ZXZ, deg):", className="small mb-0"),
                dbc.Row([dbc.Col(dcc.Slider(id='euler-phi1-slider', min=0, max=360, step=5, value=0, marks=None, tooltip={"placement": "bottom", "always_visible": False}))], className="mb-0"),
                dbc.Row([dbc.Col(dcc.Slider(id='euler-phi-slider', min=0, max=180, step=5, value=0, marks=None, tooltip={"placement": "bottom", "always_visible": False}))], className="mb-0"),
                dbc.Row([dbc.Col(dcc.Slider(id='euler-phi2-slider', min=0, max=360, step=5, value=0, marks=None, tooltip={"placement": "bottom", "always_visible": False}))], className="mb-0"),
                dbc.Row(dbc.Col(html.Div(id='euler-angles-display', className="small text-muted")))
            ]),
            html.P("(Direct dragging not implemented)", className="small text-muted mt-1 mb-2")
        ]),
        html.Hr(style={'marginTop': 5, 'marginBottom': 10}),
        dbc.Label("Pole Figure Controls", style={'fontWeight':'bold'}, className="small mb-1"),
         dbc.Row([
             dbc.Col([
                 dbc.Label("Plane Family:", html_for='pole-figure-plane-dropdown', className="small mb-1"),
                 dcc.Dropdown(id='pole-figure-plane-dropdown', value="{100}", clearable=False, className="small"),
             ], width=6),
              dbc.Col([
                  dbc.Label("Projection Axis:", html_for='projection-pole-axis-dropdown', className="small mb-1"),
                  dbc.Tooltip("Sample direction pointing out of the page (center).", target='projection-pole-axis-dropdown'),
                  dcc.Dropdown(id='projection-pole-axis-dropdown', options=[{'label': ax.upper(), 'value': ax} for ax in ['x', 'y', 'z']], value='z', clearable=False, className="small"),
              ], width=6),
         ], className="mb-2"),
         dbc.Row([
             dbc.Col([
                dbc.Label("Plot Type:", html_for='pole-figure-plot-type', className="small mb-1"),
                dcc.RadioItems(options=[{'label': ' Scatter', 'value': 'scatter'}, {'label': ' Heatmap', 'value': 'heatmap'}], value='scatter', id='pole-figure-plot-type', inline=True, labelClassName="small me-2", inputClassName="me-1"),
             ], width=12)
         ], className="mb-1"),
         html.Div(id='intensity-controls-div', style={'marginTop':'5px', 'display':'none'}, children=[
              dbc.Label("Intensity Range (MRD):", html_for='intensity-range-slider', className="small mb-1"),
              dbc.Tooltip("Adjust color scale saturation (multiples of random).", target='intensity-range-slider'),
              dcc.RangeSlider(id='intensity-range-slider', min=0, max=10, step=0.5, value=[0, 3], marks={i: str(i) for i in range(0, 11, 2)}),
         ]),
         dbc.Button("Update Pole Figure", id='update-pf-button', color='primary', size="sm", className='mt-3 w-100'),
         dbc.Tooltip("Recalculate pole figure.", target='update-pf-button'),
    ], className="p-2")
])
# endregion Control Panel Components

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Interactive Pole Figure Explorer", className="mb-3"), width=12)),
    dbc.Row([
        dbc.Col(controls_card, width=12, lg=4, className="mb-3"),
        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(id="crystal-grid-header", className="py-2"),
                    dbc.CardBody(dcc.Loading(dcc.Graph(id='crystal-grid-graph', style={'height': '60vh'})), className="p-1")
                ]), width=12, md=6, className="mb-3 mb-md-0"),
                dbc.Col(dbc.Card([
                     dbc.CardHeader(id="pole-figure-header", className="py-2"),
                     dbc.CardBody(dcc.Loading(dcc.Graph(id='pole-figure-graph', style={'height': '60vh'})), className="p-1")
                ]), width=12, md=6),
            ])
        ], width=12, lg=8),
    ]),
    dcc.Store(id='crystal-orientations-store'),
    dcc.Store(id='grid-params-store', data={'size': DEFAULT_GRID_SIZE, 'type': DEFAULT_STRUCTURE}),
    dcc.Store(id='pole-figure-data-store'),
], fluid=True)

# ==============================================================================
# 6. Callbacks
# ==============================================================================

# region Initialization and Control Updates
@app.callback(
    [Output('grid-params-store', 'data'),
     Output('pole-figure-plane-dropdown', 'options'),
     Output('pole-figure-plane-dropdown', 'value'),
     Output('slip-system-dropdown', 'options'),
     Output('slip-system-dropdown', 'value'),
     Output('crystal-grid-header', 'children'),
     Output('orientation-dist-dropdown', 'options')],  # Added new output
    [Input('grid-size-dropdown', 'value'),
     Input('crystal-type-dropdown', 'value')],
    [State('pole-figure-plane-dropdown', 'value'),
     State('slip-system-dropdown', 'value'),
     State('orientation-dist-dropdown', 'value')]  # Added new state
)
def update_grid_params_dependent_options(grid_size, crystal_type, current_pf_plane, current_slip_system, current_dist):
    grid_params = {'size': grid_size, 'type': crystal_type}
    struct_info = CRYSTAL_STRUCTURES.get(crystal_type, {})

    # Update pole figure options
    available_pf_planes = struct_info.get('pole_families', ["{100}"])
    pf_options = [{'label': p, 'value': p} for p in available_pf_planes]
    new_pf_value = available_pf_planes[0] if available_pf_planes else None
    if current_pf_plane in available_pf_planes: new_pf_value = current_pf_plane

    # Update slip system options
    available_slip_systems = struct_info.get('slip_systems', {})
    slip_options = [{'label': k, 'value': k} for k in available_slip_systems.keys()]
    # Ensure 'None' is always an option, add it if not present from data
    if 'None' not in available_slip_systems:
         slip_options.insert(0, {'label': 'None', 'value': 'None'})
    new_slip_value = 'None'
    if current_slip_system in available_slip_systems: new_slip_value = current_slip_system

    # Filter orientation distributions based on crystal type
    basic_distributions = ['Default (Aligned)', 'Random']
    filtered_textures = [k for k, v in TEXTURE_COMPONENTS.items() if v[1] == crystal_type]
    orientation_options = [{'label': d, 'value': d} for d in basic_distributions + filtered_textures]

    grid_size_label = f"{grid_size}x{grid_size}" if grid_size > 1 else "1x1"
    grid_header = f"{grid_size_label} {crystal_type} Grid"

    return grid_params, pf_options, new_pf_value, slip_options, new_slip_value, grid_header, orientation_options
    
@app.callback(
    Output('crystal-orientations-store', 'data'),
    [Input('grid-params-store', 'data'),
     Input('reset-button', 'n_clicks'),
     Input('random-button', 'n_clicks'),
     Input('orientation-dist-dropdown', 'value'),
     Input('diffusivity-slider', 'value'),
     Input('euler-phi1-slider', 'value'), Input('euler-phi-slider', 'value'), Input('euler-phi2-slider', 'value')],
    [State('crystal-orientations-store', 'data'), State('selected-crystal-index', 'value')]
)
def update_orientations(grid_params, n_reset, n_random, distribution, diffusivity, phi1, phi, phi2, current_orientations_json, selected_idx):
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No trigger'
    grid_size = grid_params['size']; crystal_type = grid_params['type']
    n_crystals = grid_size * grid_size
    orientations_obj = None # Flag to indicate if full regeneration is needed

    # --- Handle Single Crystal Rotation ---
    if trigger_id in ['euler-phi1-slider', 'euler-phi-slider', 'euler-phi2-slider'] and selected_idx >= 0 and current_orientations_json:
        try:
            quat_list = current_orientations_json['quaternions'] # Get current list
            if 0 <= selected_idx < len(quat_list):
                new_rot = R.from_euler('ZXZ', [phi1, phi, phi2], degrees=True)
                quat_list[selected_idx] = new_rot.as_quat().tolist() # Update specific element
                return {'quaternions': quat_list} # Return the modified list
            else: pass # Index out of bounds, fall through to regenerate all
        except Exception as e: print(f"Error updating single orientation: {e}") # Fall through

    # --- Handle Full Grid Regeneration ---
    regen_triggers = ['grid-params-store', 'reset-button', 'random-button', 'orientation-dist-dropdown', 'diffusivity-slider']
    # Need regeneration if triggered by regen_triggers OR if single update failed (orientations_obj still None)
    if trigger_id in regen_triggers or orientations_obj is None:
        if trigger_id == 'reset-button': orientations_obj = generate_orientations(n_crystals, 'Default (Aligned)', 0.0, crystal_type)
        elif trigger_id == 'random-button': orientations_obj = generate_orientations(n_crystals, 'Random', 0.0, crystal_type)
        else: orientations_obj = generate_orientations(n_crystals, distribution, diffusivity, crystal_type)

        if orientations_obj is not None:
            if orientations_obj.single: quat_list = [orientations_obj.as_quat().tolist()]
            else: quat_list = [q.tolist() for q in orientations_obj.as_quat()]
            return {'quaternions': quat_list}
        else: return no_update # Should not happen if generation works

    # If trigger wasn't regeneration or single update didn't happen/succeed, do nothing
    return no_update
# endregion

# region Individual Crystal Selection/Manipulation Callbacks
@app.callback(
    [Output('selected-crystal-index', 'value'), Output('euler-sliders-div', 'style'),
     Output('euler-phi1-slider', 'value'), Output('euler-phi-slider', 'value'), Output('euler-phi2-slider', 'value'),
     Output('euler-angles-display', 'children')],
    [Input('crystal-grid-graph', 'clickData')],
    [State('selected-crystal-index', 'value'), State('crystal-orientations-store', 'data')],
    prevent_initial_call=True
)
def display_click_data_and_update_sliders(clickData, current_selected_idx, orientation_data):
    if not clickData: return no_update, no_update, no_update, no_update, no_update, no_update
    point = clickData['points'][0]
    if 'customdata' in point:
        selected_idx = point['customdata'][0]
        if orientation_data and 'quaternions' in orientation_data and 0 <= selected_idx < len(orientation_data['quaternions']):
             try:
                 quat = orientation_data['quaternions'][selected_idx]
                 rot = R.from_quat(quat); euler_angles = rot.as_euler('ZXZ', degrees=True)
                 phi1, phi, phi2 = euler_angles[0], euler_angles[1], euler_angles[2]
                 style = {'display': 'block', 'marginTop':'5px'}
                 angle_text = f"φ1:{phi1:.1f} Φ:{phi:.1f} φ2:{phi2:.1f}"
                 return selected_idx, style, phi1, phi, phi2, angle_text
             except Exception as e:
                 print(f"Error getting Euler angles for crystal {selected_idx}: {e}")
                 style = {'display': 'block', 'marginTop':'5px'}; return selected_idx, style, 0,0,0, "Error"
        elif selected_idx == current_selected_idx:
             return no_update, no_update, no_update, no_update, no_update, no_update
        else:
             style = {'display': 'block', 'marginTop':'5px'}; return selected_idx, style, 0, 0, 0, "Data N/A"
    style = {'display': 'none'}; return -1, style, 0, 0, 0, ""

@app.callback( Output('euler-angles-display', 'children', allow_duplicate=True),
    [Input('euler-phi1-slider', 'value'), Input('euler-phi-slider', 'value'), Input('euler-phi2-slider', 'value')],
    [State('selected-crystal-index', 'value')], prevent_initial_call=True )
def update_euler_display_from_sliders(phi1, phi, phi2, selected_idx):
    if selected_idx >= 0: return f"φ1:{phi1:.1f} Φ:{phi:.1f} φ2:{phi2:.1f}"
    return ""
# endregion

# region Visualization Callbacks
@app.callback(
    Output('crystal-grid-graph', 'figure'),
    [Input('crystal-orientations-store', 'data'),
     Input('grid-params-store', 'data'),
     Input('slip-system-dropdown', 'value')],
    prevent_initial_call=True
)
def update_crystal_grid_visualization(orientation_data, grid_params, selected_slip_system):
    # --- Start of Function (error checking, getting orientations etc.) ---
    if not orientation_data or not grid_params: return go.Figure()
    grid_size = grid_params['size']; crystal_type = grid_params['type']
    struct_info = CRYSTAL_STRUCTURES.get(crystal_type, {})
    shape_type = struct_info.get('shape', 'cube')
    c_a_ratio = struct_info.get('c_a', HCP_CA_RATIO) if shape_type == 'hex' else None

    try:
        quat_list = orientation_data['quaternions']
        n_expected = grid_size * grid_size
        if len(quat_list) != n_expected: raise ValueError(f"Expected {n_expected} orientations, got {len(quat_list)}")
        orientations = R.from_quat(quat_list)
    except Exception as e: print(f"Error processing grid orientation data: {e}"); return go.Figure(layout=go.Layout(title="Error Loading Orientations"))

    meshes = []; crystal_render_size = DEFAULT_CRYSTAL_SIZE * (1.0 - GAP_FRACTION)
    cell_size = 1.0; slip_plane_size = crystal_render_size * 0.95

    # --- Get slip planes ---
    slip_planes_to_draw = [] # Default to empty list
    if selected_slip_system != "None":
         slip_data = struct_info.get("slip_systems", {}).get(selected_slip_system, {})
         # Ensure the fetched value is treated correctly, default to empty list if key missing
         slip_planes_to_draw = slip_data.get("planes", [])

    # --- Loop through crystals ---
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            center = np.array([(c-(grid_size-1)/2.0)*cell_size, (r-(grid_size-1)/2.0)*cell_size, 0])
            rotation = orientations[idx] if not orientations.single else orientations

            # --- Create Crystal Mesh ---
            if shape_type == 'cube': base_mesh = get_cube_mesh(size=crystal_render_size, name=f'C {idx}')
            elif shape_type == 'hex': base_mesh = get_hex_prism_mesh(size=crystal_render_size, height_factor=c_a_ratio, name=f'C {idx}')
            else: base_mesh = get_cube_mesh(size=crystal_render_size, name=f'C {idx}')
            vertices = np.vstack([base_mesh.x, base_mesh.y, base_mesh.z]).T
            translated_vertices = rotation.apply(vertices) + center
            meshes.append(go.Mesh3d(x=translated_vertices[:,0],y=translated_vertices[:,1],z=translated_vertices[:,2],
                i=base_mesh.i, j=base_mesh.j, k=base_mesh.k, color='cornflowerblue', opacity=0.85,
                flatshading=True, name=f'C {idx}', customdata=[idx], hoverinfo='name'))

            # --- Add Slip Planes ---
            if isinstance(slip_planes_to_draw, np.ndarray) and slip_planes_to_draw.size > 0:
                plane_color = 'rgba(230, 100, 100, 0.6)'
                rotated_normals = rotation.apply(slip_planes_to_draw)
                for i_plane, rotated_normal in enumerate(rotated_normals):
                    # Add robustness check inside get_plane_mesh or here
                    if np.linalg.norm(rotated_normal) > 1e-6:
                         plane_mesh = get_plane_mesh(rotated_normal, center, size=slip_plane_size, color=plane_color, name=f'Slip {idx}-{i_plane}')
                         meshes.append(plane_mesh)

            idx += 1

    # --- Add Improved Axes and Finalize Figure ---
    axis_len = grid_size * cell_size * 0.6 if grid_size > 0 else 0.6
    max_coord = max(1.0, grid_size * cell_size / 2 * 1.2)
    axis_range = [-max_coord, max_coord]
    
    # Use the improved axes function
    axes_elements = create_improved_axes(axis_len)
    meshes.extend(axes_elements)
    
    scene_layout = dict(
        xaxis=dict(visible=True, showticklabels=False, title='', range=axis_range, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
        yaxis=dict(visible=True, showticklabels=False, title='', range=axis_range, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
        zaxis=dict(visible=True, showticklabels=False, title='', range=axis_range, showgrid=False, zeroline=False, backgroundcolor="rgba(0,0,0,0)"),
        aspectmode='data', camera=dict(eye=dict(x=1.8, y=1.8, z=0.8))
    )
    
    fig = go.Figure(data=meshes, layout=go.Layout(scene=scene_layout, margin=dict(l=5, r=5, t=5, b=5)))
    fig.update_layout(clickmode='event+select')
    return fig

@app.callback(
    Output('pole-figure-data-store', 'data'),
    [Input('update-pf-button', 'n_clicks')],
    [State('crystal-orientations-store', 'data'), State('grid-params-store', 'data'),
     State('pole-figure-plane-dropdown', 'value'), State('projection-pole-axis-dropdown', 'value')]
)
def calculate_pole_figure_data(n_clicks, orientation_data, grid_params, plane_family, projection_axis):
    if not orientation_data or not grid_params or not plane_family: return None
    crystal_type = grid_params['type']
    c_a_ratio = CRYSTAL_STRUCTURES.get(crystal_type,{}).get('c_a', HCP_CA_RATIO)
    grid_size = grid_params['size']; n_crystals = grid_size*grid_size
    try:
        quat_list = orientation_data['quaternions']
        if len(quat_list) != n_crystals: raise ValueError("Orientation data mismatch")
        orientations = R.from_quat(quat_list)
    except Exception as e: print(f"Error processing orientation data for PF calc: {e}"); return None
    base_normals = get_plane_family_normals(plane_family, crystal_type, c_a_ratio)
    if not isinstance(base_normals, np.ndarray) or base_normals.size == 0: print(f"Could not get valid normals for {plane_family} ({crystal_type})"); return None
    all_rotated_normals = []
    orientations_list = [orientations] if orientations.single else orientations
    for i, rot in enumerate(orientations_list):
        if not isinstance(rot, R) or not rot.single: continue
        try:
            # Ensure base_normals is not empty before applying
            if base_normals.size > 0:
                 rotated_for_this_crystal = rot.apply(base_normals)
                 all_rotated_normals.append(rotated_for_this_crystal)
        except ValueError as e: print(f"ERROR during rot.apply() for crystal index {i}: {e}"); continue
    if not all_rotated_normals: return {'points': []}
    all_rotated_normals = np.concatenate(all_rotated_normals, axis=0)
    projected_points = stereographic_projection(all_rotated_normals, projection_axis)
    return {'points': projected_points.tolist() if projected_points.size > 0 else []}

@app.callback(
    [Output('pole-figure-graph', 'figure'), Output('pole-figure-header', 'children'), Output('intensity-controls-div', 'style')],
    [Input('pole-figure-data-store', 'data')],
    [State('pole-figure-plane-dropdown', 'value'), State('projection-pole-axis-dropdown', 'value'),
     State('pole-figure-plot-type', 'value'), State('intensity-range-slider', 'value'), State('grid-params-store', 'data')]
)
def render_pole_figure(pf_data, plane_family, projection_axis, plot_type, intensity_range, grid_params):
    plane_family_label = plane_family if plane_family else "N/A"
    title = f"PF ({plane_family_label}) - {projection_axis.upper()} Proj."
    fig = go.Figure()
    intensity_controls_style = {'marginTop':'5px', 'display':'none'}
    
    # Determine proper axis labels based on projection axis
    if projection_axis == 'z':
        x_label = 'X'  # Sample X direction
        y_label = 'Y'  # Sample Y direction
    elif projection_axis == 'x':
        x_label = 'Y'  # Sample Y direction
        y_label = 'Z'  # Sample Z direction
    elif projection_axis == 'y':
        x_label = 'X'  # Sample X direction
        y_label = 'Z'  # Sample Z direction
    
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=16, color='black')), 
            range=[-1.3, 1.3], 
            constrain='domain', 
            showgrid=False, 
            zeroline=True,
            showticklabels=True
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=16, color='black')), 
            range=[-1.3, 1.3], 
            scaleanchor='x', 
            showgrid=False, 
            zeroline=True,
            showticklabels=True
        ),
        showlegend=True, 
        margin=dict(l=20, r=20, t=40, b=20), 
        plot_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Add circle for stereographic projection boundary
    angles = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(
        x=np.cos(angles), 
        y=np.sin(angles), 
        mode='lines', 
        line=dict(color='darkgrey', width=1.5), 
        name='Projection Boundary', 
        hoverinfo='none'
    ))

    if pf_data and pf_data.get('points'):
        projected_points = np.array(pf_data['points'])
        if projected_points.ndim == 2 and projected_points.shape[0] > 0:
             if plot_type == 'scatter':
                 fig.add_trace(go.Scatter(x=projected_points[:, 0], y=projected_points[:, 1], mode='markers', marker=dict(size=5, color='#e41a1c', opacity=0.7), name='Poles'))
             elif plot_type == 'heatmap':
                 intensity_controls_style = {'marginTop':'5px', 'display':'block'}
                 n_bins = 64; hist_range = [[-1.05, 1.05], [-1.05, 1.05]]
                 if projected_points.ndim == 2 and projected_points.shape[1] == 2:
                     H, xedges, yedges = np.histogram2d(projected_points[:, 0], projected_points[:, 1], bins=n_bins, range=hist_range)
                     xcenters=(xedges[:-1]+xedges[1:])/2; ycenters=(yedges[:-1]+yedges[1:])/2
                     X,Y=np.meshgrid(xcenters, ycenters); mask=X**2+Y**2>1.0; H[mask.T]=np.nan
                     valid_H = H[~np.isnan(H)]; mrd_unit = valid_H.mean() if valid_H.size > 0 and valid_H.mean() > 1e-6 else 1.0
                     H_mrd = H / mrd_unit
                     min_intensity, max_intensity = intensity_range
                     fig.add_trace(go.Heatmap(z=H_mrd, x=xcenters, y=ycenters, colorscale='viridis', zmin=min_intensity, zmax=max_intensity, colorbar=dict(title='MRD', thickness=15, len=0.7), hoverinfo='z'))
                 else: print("Warning: Projected points data has unexpected shape for heatmap.")
    
    return fig, title, intensity_controls_style
# endregion Visualization Callbacks

# ==============================================================================
# 7. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    app.run(debug=True, port=8050)

# --- END OF FILE ---