import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Rocket Motor Test Viewer", page_icon="🚀")
st.title("Rocket Motor Test Viewer")

# Allow users to add multiple new fields dynamically



type_units = {
            "Time": ["s", "ms", "us"],
            "Pressure": ["Pa","MPa","kPa", "bar", "psi"],
            "Temperature": ["C", "F", "K"],
            "Force": ["N", "kN", "kgf", "lbf"]
        }

class Filter_type:
    def Smoothing(df:pd.DataFrame,target:str, factor:float):
        assert 0 < factor< 100, "Smoothing factor must be greater than 0 and less than 0.1"
        dfc = df.copy()
        n = int(len(dfc[target])*factor)
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        dfc[target] = dfc[target].rolling(window=n, min_periods=1).mean()
        return dfc
    
    def Threshold(df:pd.DataFrame, target:str, threshold:float):
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        dfc = df.copy()
        return dfc[target].apply(lambda x: x if x < threshold else threshold)

    def Frequency(df:pd.DataFrame, target:str, frequency:int):
        dfc = df.copy()
        if target not in dfc.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame")
        signal = dfc[target].values
        time = dfc.index.values
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal), d=np.mean(np.diff(time)))
        fft[np.abs(freqs - frequency) < 0.2*frequency] = 0
        fft[np.abs(freqs + frequency) < 0.2*frequency] = 0
        filtered_signal = np.fft.ifft(fft).real
        dfc[target] = filtered_signal
        return dfc
            

def load_csv():
    csv = st.file_uploader("Upload a CSV with test data", type=["csv"])
    if csv is not None: 
        try:
            st.session_state.raw_data = pd.read_csv(csv)
            st.write("Input Data:")
            st.dataframe(df)
            
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    

def add_field(fields:list, columns):
    fields.append({"name": "", "column": columns[2], "type": "", "unit": "", "smoothing": ""})
def add_filter(filters:list):
    filters.append({"targets": [], "type": Filter_type.Smoothing, "arg": 0.1})

def render_Base_Axis():
    BA = st.session_state.df_config['Base_Axis']
    for axis in BA.keys():
        col1, col2 = st.columns([3,2])
        with col1:
            BA[axis]['column'] = st.selectbox(
                f"{axis} column", options=df.columns.tolist(), key=f"{axis}_column"
            )
        with col2:
            BA[axis]['unit'] = st.selectbox(f"{axis} Unit", 
                                options=type_units['Time'] if axis == 'Time' else type_units['Force'], 
                                key=f"{axis}_unit")
        


def render_Fields():
    st.divider()
    st.write("Additional Fields:")
    for i, field in enumerate(Fields):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            Fields[i]["name"] = st.text_input(
                f"Name", value=field["name"], key=f"field_name_{i}"
            )
        with col2:
            Fields[i]["column"] = st.selectbox(
                f"Column", options=df.columns.tolist(),
                index=df.columns.get_loc(field["column"]) if field["column"] in df.columns else 0,
                key=f"field_column_{i}"
            )
        with col3:
            Fields[i]["type"] = st.selectbox(
                f"Type", options=["Pressure", "Temperature", "Force"], key=f"field_type_{i}"
            )
        with col4:
            units_options = type_units.get(Fields[i]["type"], [""])
            Fields[i]["unit"] = st.selectbox(
                f"Unit", options=units_options, 
                index=units_options.index(field["unit"]) if field["unit"] in units_options else 0,
                key=f"field_unit_{i}"
            )
        with col5:
            st.markdown("<div style='height:1.75em;'>Delete</div>", unsafe_allow_html=True)
            if st.button("🗑️", key=f"remove_field_{i}"):
                Fields.pop(i)
                st.rerun()

def render_Filters():
    st.divider()
    st.write("Filters:")
    filter_methods = [func for func in dir(Filter_type) if callable(getattr(Filter_type, func)) and not func.startswith("__")]
    for i, filter in enumerate(Filters):
            col1, col2, col3, col4= st.columns(4)
            with col1:
                Filters[i]["targets"] = st.multiselect(
                    f"Filter Targets", options=df.columns.tolist(), default=filter["targets"],
                    key=f"filter_targets_{i}"
                    )
            with col2:
                selected_method = st.selectbox(
                    f"Filter Type", options=filter_methods, key=f"filter_type_{i}"
                    )
                Filters[i]["type"] = getattr(Filter_type, selected_method)
            with col3:
                Filters[i]["arg"] = st.number_input(
                    f"{Filters[i]["type"].__code__.co_varnames[2].capitalize()}",
                    min_value = 0.001, key=f"filter_arg_{i}"
                    )
            with col4:
                st.markdown("<div style='height:1.75em;'>Delete</div>", unsafe_allow_html=True)
                if st.button("🗑️", key=f"remove_filter_{i}"):
                    Filters.pop(i)
                    st.rerun()

def render_buttons():
    st.divider()
    col1, col2, col3= st.columns(3)
    with col1:
        if st.button("Add Field"):
            add_field(Fields, df.columns.tolist())
            st.rerun()
    with col2:
        if st.button("Add Filter"):
            add_filter(Filters)
            st.rerun()
    plot_clicked = col3.button("Plot Data")
    if plot_clicked:
        process_data()
        st.rerun()

def process_data():
    columns =   [v["column"] for v in st.session_state.df_config['Base_Axis'].values() if v["column"] is not None]
    columns +=  [field["column"] for field in st.session_state.df_config['Fields'] if field.get("column") is not None]

    dfc = df[columns].copy()
    
    # Convert units to SI for additional Fields
    for field in st.session_state.df_config['Fields'] + list(st.session_state.df_config['Base_Axis'].values()):
        col = field.get("column")
        unit = field.get("unit")
        match unit:
            case "MPa": dfc[col] = dfc[col] * 1e6
            case "kPa": dfc[col] = dfc[col] * 1e3
            case "bar": dfc[col] = dfc[col] * 1e5
            case "psi": dfc[col] = dfc[col] * 6_894.76
            case "F": dfc[col] = (dfc[col] - 32) * 5.0/9.0
            case "K": dfc[col] = dfc[col] - 273.15
            case "kN": dfc[col] = dfc[col] * 1e3
            case "kgf": dfc[col] = dfc[col] * 9.80665
            case "lbf": dfc[col] = dfc[col] * 4.44822
            case "ms": dfc[col] = dfc[col] * 1e-3
            case "us": dfc[col] = dfc[col] * 1e-6
            case _: pass

    dfc.columns = ["Time (s)", "Thrust (N)"] + [
        f"{field['name']} ({type_units.get(field['type'], [''])[0]})"
        for field in st.session_state.df_config['Fields']   
    ]

    max_thrust = dfc['Thrust (N)'].max()
    threshold = 0.01 * max_thrust
    dfc = dfc[dfc['Thrust (N)'] >= threshold]

    dfc['Time (s)'] = dfc['Time (s)'] - dfc['Time (s)'].min()
    dfc.set_index('Time (s)', inplace=True)
    
    for filter in st.session_state.df_config['Filters']:
        for tgt in filter["targets"]:
            field = next((f for f in st.session_state.df_config['Fields'] if f["column"] == tgt), None)
            tgt_new_name = f"{field['name']} ({type_units.get(field['type'], [''])[0]})"
            dfc = filter["type"](dfc, tgt_new_name, filter["arg"])

    st.session_state.processed_data = dfc


def render_processed_data():
    st.divider()
    st.write("Processed Data:")
    dfc = st.session_state.processed_data
    y_columns = dfc.columns.tolist()
    fig = go.Figure()
    # First y-axis (left)
    fig.add_trace(go.Scatter(
        x=dfc.index,
        y=dfc[y_columns[0]],
        name=y_columns[0],
        yaxis="y1"
    ))
    # Additional y-axes (right)
    for i, col in enumerate(y_columns[1:], start=2):
        fig.add_trace(go.Scatter(
            x=dfc.index,
            y=dfc[col],
            name=col,
            yaxis=f"y{i}"
        ))
    # Layout for multiple y-axes
    layout = dict(
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title=y_columns[0]),
    )
    for i, col in enumerate(y_columns[1:], start=2):
        layout[f"yaxis{i}"] = dict(
            title=col,
            anchor="free",
            overlaying="y",
            side="right",
            position=1.0 - 0.05 * (i-2   )
        )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(dfc)
            

if "df_config" not in st.session_state:
    st.session_state.df_config = {
        "Base_Axis":{
            "Time": {
                "column": None,
                "unit": None
            },
            "Thrust": {
                "column": None,
                "unit": None,
                "smoothng": None
            }
        },
        "Fields": [],
        "Filters": []
    }

if "processed_data" not in st.session_state:
            st.session_state.processed_data = pd.DataFrame()
if "raw_data" not in st.session_state:
            st.session_state.raw_data = pd.DataFrame()

df = st.session_state.raw_data 
load_csv()
render_Base_Axis()
Fields = st.session_state.df_config['Fields']
render_Fields()
Filters = st.session_state.df_config['Filters']
render_Filters()
render_buttons()
if not st.session_state.processed_data.empty:
    render_processed_data()