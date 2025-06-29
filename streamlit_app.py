import streamlit as st
import os
import pandas as pd
import requests
import openpyxl
from io import BytesIO
from zipfile import ZipFile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from probability_matrix import GetMatrix,ProbabilityMatrix
import custom_filtering_dataframe
from returns_main import folder_input,folder_processed_pq
import requests
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc  # For memory management
import psutil  # For memory monitoring
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats
from sklearn.neighbors import KernelDensity

# Setting up page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="FR Live Plots",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Memory monitoring function
def check_memory_and_cleanup():
    """Check memory usage and force cleanup if approaching limit"""
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    
    if mem_mb > 2000:  # 2GB hard limit
        # Force aggressive cleanup
        plt.close('all')  # Close all matplotlib figures
        gc.collect()
        
        # If still over limit, clear some caches
        if process.memory_info().rss / 1024 / 1024 > 2000:
            st.cache_data.clear()
            gc.collect()
            st.warning("Memory limit reached. Cleared caches to prevent crash.")
    
    return mem_mb

# Defining custom functions to modify generated data as per user input
def get_volatility_returns_csv_stats_custom_days(target_csv,target_column):
        
    stats_csv=target_csv[target_column].describe(percentiles=[0.1,0.25,0.5,0.75,0.95,0.99])
    # Add additional statistics to the DataFrame
    stats_csv.loc['mean'] = target_csv[target_column].mean()
    stats_csv.loc['skewness'] = target_csv[target_column].skew()
    stats_csv.loc['kurtosis'] = target_csv[target_column].kurtosis()

    stats_csv.index.name = 'Volatility of Returns Statistic'
    return stats_csv

def get_volatility_returns_csv_custom_days(target_csv,target_column):
    target_csv['ZScore wrt Given Days']=(target_csv[target_column]-target_csv[target_column].mean())/target_csv[target_column].std()
    return target_csv

# Defining functions to download the data

# 1. Function to convert DataFrame to Excel file with multiple sheets
def download_combined_excel(df_list,sheet_names,skip_index_sheet=[]):
    # Create a BytesIO object to hold the Excel file
    output = BytesIO()

    # Create a Pandas Excel writer using openpyxl as the engine
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheetname,mydf in zip(sheet_names,df_list):
            if sheetname in skip_index_sheet:
                mydf.to_excel(writer, sheet_name=sheetname,index=False)
            else:
                mydf.to_excel(writer, sheet_name=sheetname)
    # Save the Excel file to the BytesIO object
    output.seek(0)
    return output


# 2. Main function to read image url and download as png files
def process_images(image_url_list):
    # Logic for downloading image bytes
    st.session_state["image_bytes_list"] = get_image_bytes(image_url_list)
    st.session_state["button_clicked"] = False  # Reset the button state after processing is complete

# 2.1 Function to get image bytes from list of images.
def get_image_bytes(image_url_list):
    image_bytes = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_image, image_url_list)
        for result in results:
            if result:
                image_bytes.append(result)
    return image_bytes

# 2.2 Function to fetch image url
def fetch_image(url):
    try:
        response = requests.get(url, timeout=10)  # Add a timeout to prevent hanging
        response.raise_for_status()  # Raise HTTP errors if any
        image = Image.open(BytesIO(response.content))  # Open the image
        output = BytesIO()
        image.save(output, format='PNG')  # Save the image in PNG format
        output.seek(0)
        return output
    except Exception as e:
        st.error(f"Error processing image {url}: {e}")
        return None
    
# 2.3 Function to download image created via matplotlib.
def download_img_via_matplotlib(plt_object):
    buf=BytesIO()
    plt_object.savefig(buf, format="png",bbox_inches='tight')
    buf.seek(0)  # Go to the beginning of the buffer
    return buf

# 3. Function to create a ZIP file (not used)
def create_zip(excel_file_list, image_bytes_list):
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, 'w') as zip_file:
        # Add Excel file
        for excel_file in excel_file_list:
            zip_file.writestr('combined_data.xlsx', excel_file.getvalue())
        # Add image file
        for image_bytes in image_bytes_list:
            zip_file.writestr('example_image.png', image_bytes.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# 4. Cached function to load CSV from URL
@st.cache_data
def load_csv_from_url(url, index_col=None):
    """Load CSV from URL with caching to avoid repeated downloads"""
    if index_col is not None:
        return pd.read_csv(url, index_col=index_col)
    return pd.read_csv(url)

#5.0 helper function for 5.1
def add_start_end_ts(all_event_ts , delta):

    if(delta < 0):  # pre event + custom with delta < 0

        all_event_ts['end'] = all_event_ts['timestamp'].apply(lambda x: x.replace(minute=0, second=0, microsecond=0)) 
        all_event_ts['start'] = all_event_ts['end'] + pd.Timedelta(hours = delta)

    else:   # immediate reaction + custom with delta > 0

        all_event_ts['start'] = all_event_ts['timestamp'].apply(lambda x: x.replace(minute=0, second=0, microsecond=0))
        all_event_ts['end'] = all_event_ts['start'] + pd.Timedelta(hours = delta)

    return all_event_ts
    
#5.1 calculating the returns for event specific distros
@st.cache_data
def calc_event_spec_returns(selected_event , all_event_ts , ohcl_1h , mode , event_list, delta = 0, filter_out_other_events=False,  time_gap_hours=2):
    
    # Work with views instead of copies when possible to save memory
    event_ts = all_event_ts
    
    # Only create a copy if we need to modify the data
    if 'events' in event_ts.columns:
        # Check if we need to convert to string
        if event_ts['events'].dtype != 'object':
            event_ts = event_ts.copy()
        event_ts['events'] = event_ts['events'].astype(str)


    ############## Added by Yaman #######################################################################################################

    event_list_lower = [e.strip().lower() for e in event_list]

    def pick_event(x):
        x_l = x.lower()
        for e in event_list_lower:
            if e in x_l:
                return e
        return None

    event_ts['events'] = event_ts['events'].apply(pick_event)

    event_ts = event_ts.dropna(subset=['events'])

    event_ts = event_ts.drop_duplicates(subset=['timestamp','events'], keep='first')

    cutoff_time = pd.to_datetime('2022-12-20 00:00:00-05:00', errors='coerce')
    event_ts = event_ts[event_ts['timestamp'] >= cutoff_time]


    event_ts_filtered = event_ts.loc[event_ts['events'].str.strip().str.lower().str.contains(selected_event , case=False, na=False)]

    if filter_out_other_events:
            
            event_list_lower = [e.strip().lower() for e in event_list if e.strip().lower() != selected_event.lower()]

            clean_rows = []
            counter1 = 0
            counter2 = 0
            for _, row in event_ts_filtered.iterrows():
                t = row['timestamp']
                t_minus = t - pd.Timedelta(hours=time_gap_hours)
                t_plus = t + pd.Timedelta(hours=time_gap_hours)

                # Get full window, including current row
                nearby_events = event_ts[
                    (event_ts['timestamp'] >= t_minus) &
                    (event_ts['timestamp'] <= t_plus)
                ]

                # Now: Check if any disallowed event appears in the entire window
                contains_unwanted_event = (
                    nearby_events['events']
                    .str.lower()
                    .apply(lambda x: any(e in x for e in event_list_lower))
                    .any()
                )

                if not contains_unwanted_event:
                    clean_rows.append(row)
                    counter1 += 1
                else:
                    counter2 += 1

            print(f"Out of {counter1+counter2} times we see this event, only {counter2} times do we see another major event within +- 2 hours interval of it.")

            # Keep only rows with no unwanted overlap
            event_ts_filtered = pd.DataFrame(clean_rows)

    event_ts = event_ts_filtered     #Setting event_ts to event_ts_filtered so that the rest of the code below works as it was.

    ########################################################################################################################################

    #pre event
    if(mode == 1):
        event_ts = add_start_end_ts(event_ts , -8)

    #during event (may have to be changed for future events)
    elif(mode == 2):
        event_ts = add_start_end_ts(event_ts , 1)

    #custom (delta will be non-zero in this case)
    else:
        event_ts = add_start_end_ts(event_ts , delta)  

    event_ts = event_ts.drop_duplicates(subset=['start'], keep='first')
    cutoff_time = pd.to_datetime('2022-12-20 00:00:00-05:00', errors='coerce')
    event_ts = event_ts[event_ts['start'] >= cutoff_time]

    # Check memory before processing
    check_memory_and_cleanup()
    
    # Pre-allocate numpy arrays for better memory efficiency
    n_events = len(event_ts)
    
    # For large event sets (like NFP), process in smaller chunks
    if n_events > 100:
        # st.info(f"Processing {n_events} events. This may take a moment...")
        batch_size = 25  # Smaller batches for large datasets
        
        # For very large datasets, use even smaller batches to prevent memory spikes
        if n_events > 200:
            batch_size = 10
            # st.info("Using optimized processing for large dataset...")
    else:
        batch_size = 50
    
    vol_ret = np.empty(n_events, dtype=np.float32)
    abs_ret = np.empty(n_events, dtype=np.float32)
    ret = np.empty(n_events, dtype=np.float32)
    start_date = []
    end_date = []
    
    for i, (end, start) in enumerate(zip(event_ts['end'], event_ts['start'])):
        # More efficient filtering using index slicing if possible
        mask = (ohcl_1h['US/Eastern Timezone'] >= start) & (ohcl_1h['US/Eastern Timezone'] < end)
        temp_df = ohcl_1h.loc[mask]
        
        if temp_df.empty:
            vol_ret[i] = np.nan
            abs_ret[i] = np.nan
            ret[i] = np.nan
            start_date.append(np.nan)
            end_date.append(np.nan)
        else:
            # Use numpy operations directly for speed
            high_val = temp_df['High'].values.max()
            low_val = temp_df['Low'].values.min()
            close_val = temp_df['Close'].iloc[-1]
            open_val = temp_df['Open'].iloc[0]
            
            vol_ret[i] = (high_val - low_val) * 16
            abs_ret[i] = abs(close_val - open_val) * 16
            ret[i] = (close_val - open_val) * 16
            start_date.append(temp_df['US/Eastern Timezone'].iloc[0])
            end_date.append(temp_df['US/Eastern Timezone'].iloc[-1])
        
        # Clean up memory every batch_size iterations
        if (i + 1) % batch_size == 0:
            # Check memory usage mid-processing
            current_mem = psutil.Process().memory_info().rss / 1024 / 1024
            if current_mem > 1800:  # Getting close to 2GB limit
                # st.warning(f"High memory usage detected ({current_mem:.0f}MB). Cleaning up...")
                plt.close('all')
                gc.collect()
                
                # If still high after cleanup, reduce batch size
                if psutil.Process().memory_info().rss / 1024 / 1024 > 1800:
                    batch_size = max(10, batch_size // 2)
                    # st.info(f"Reduced batch size to {batch_size} to prevent crashes")
    
    # Create DataFrame from arrays
    final_df = pd.DataFrame({
        'Volatility Return': vol_ret,
        'Absolute Return': abs_ret,
        'Return': ret,
        'Start_Date': start_date,
        'End_Date': end_date
    })
    
    # Remove NaN values
    final_df.dropna(inplace=True)
    
    # Final memory cleanup
    gc.collect()

    print("SELECTED EVENT: ", selected_event)
    print('No of Data points: ' , len(final_df))
    # print(final_df)

    return final_df
        
#5.2 plot the event specific returns using Plotly (client-side rendering)
def plot_event_spec_returns(final_df , selected_event , dur):
    # Check memory before plotting
    check_memory_and_cleanup()
    
    # Create subplots for the three distributions
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Absolute Return", "Return", "Volatility Return"],
        horizontal_spacing=0.1
    )
    
    col_idx = 0
    for col in final_df.columns[:3]:
        col_idx += 1
        
        # Get data for this column
        data = final_df[col].dropna()
        
        # Statistics
        stats = data.describe()
        mean = stats['mean']
        std = stats['std']
        current_value = data.iloc[-1]
        current_date = final_df['Start_Date'].iloc[-1].date()
        zscore = (current_value - mean) / std if std != 0 else 0
        
        # Create histogram
        hist, bin_edges = np.histogram(data, bins='auto', density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Add histogram bars
        fig.add_trace(
            go.Bar(
                x=bin_centers,
                y=hist,
                name='',
                marker_color='skyblue',
                width=bin_edges[1] - bin_edges[0],
                showlegend=False,
                text=[f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(hist))],
                textposition='outside',
                textfont=dict(size=7),
                hovertemplate='%{text}<br>Density: %{y:.3f}<extra></extra>'
            ),
            row=1, col=col_idx
        )
        
        # Calculate KDE for smooth curve
        # Extend the range beyond data min/max to match matplotlib behavior
        data_range = data.max() - data.min()
        kde_padding = data_range * 0.2  # Add 20% padding on each side
        kde_x = np.linspace(data.min() - kde_padding, data.max() + kde_padding, 200)
        kde = scipy_stats.gaussian_kde(data)
        kde_y = kde(kde_x)
        
        # Add KDE curve
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y,
                mode='lines',
                name='',
                line=dict(color='darkblue', width=2),
                showlegend=False,
                hovertemplate='Value: %{x:.2f}<br>Density: %{y:.3f}<extra></extra>'
            ),
            row=1, col=col_idx
        )
        
        # Add vertical line at current value
        fig.add_vline(
            x=current_value,
            line=dict(color="red", width=1, dash="dot"),
            row=1, col=col_idx
        )
        
        # Add red dot just above x-axis
        y_max = max(max(hist), max(kde_y))
        dot_y = y_max * 0.02
        
        fig.add_trace(
            go.Scatter(
                x=[current_value],
                y=[dot_y],
                mode='markers',
                marker=dict(color='red', size=8),
                showlegend=False,
                hovertemplate=f'Current Value<br>Value: {current_value:.2f}<br>Z-Score: {zscore:.2f}<br>Date: {current_date}<extra></extra>'
            ),
            row=1, col=col_idx
        )
        
        # Define xref and yref based on column index
        if col_idx == 1:
            xref = "x"
            yref = "y"
            xref_domain = "x domain"
            yref_domain = "y domain"
        else:
            xref = f"x{col_idx}"
            yref = f"y{col_idx}"
            xref_domain = f"x{col_idx} domain"
            yref_domain = f"y{col_idx} domain"
        
        # Add annotation with arrow
        fig.add_annotation(
            x=current_value,
            y=y_max * 0.15,
            text=f"Value: {current_value:.2f}, Z: {zscore:.2f}, Date: {current_date}",
            showarrow=True,
            arrowhead=2,
            arrowcolor='red',
            arrowwidth=1.5,
            ax=0,
            ay=-30,
            font=dict(color='red', size=11),
            xref=xref,
            yref=yref
        )
        
        # Add statistics box
        stats_text = (
            f"Mean: {mean:.2f}<br>"
            f"Std: {std:.2f}<br>"
            f"Min: {stats['min']:.2f}<br>"
            f"25%: {stats['25%']:.2f}<br>"
            f"Median: {stats['50%']:.2f}<br>"
            f"75%: {stats['75%']:.2f}<br>"
            f"95%: {data.quantile(0.95):.2f}<br>"
            f"99%: {data.quantile(0.99):.2f}<br>"
            f"Max: {stats['max']:.2f}"
        )
        
        fig.add_annotation(
            text=stats_text,
            xref=xref_domain,
            yref=yref_domain,
            x=0.95,
            y=0.95,
            xanchor='right',
            yanchor='top',
            showarrow=False,
            bordercolor='black',
            borderwidth=1,
            bgcolor='white',
            font=dict(size=10)
        )
        
        # Update axes
        fig.update_xaxes(title_text="Value", row=1, col=col_idx)
        if col_idx == 1:
            fig.update_yaxes(title_text="Density", row=1, col=col_idx)
        
        # Set y-axis range with padding to match matplotlib
        y_padding = y_max * 0.1
        fig.update_yaxes(range=[0, y_max + y_padding], row=1, col=col_idx)
    
    # Update layout
    fig.update_layout(
        height=400,
        showlegend=False,
        title={
            'text': 'Distribution Analysis',
            'x': 0,
            'xanchor': 'left',
            'font': {'size': 24}
        }
    )
    
    # Display the figure
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanations below
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("**Absolute Return = [abs(close-open)]**")
    with col2:
        st.write("**Return = [close - open]**")
    with col3:
        st.write("**Volatility Return = [high - low]**")
    
    # Clean up memory after plotting
    del fig
    gc.collect()

    
# Setting up tabs
tab1, tab2, tab3,tab4,tab5 = st.tabs(["Session and Volatility Returns for all sessions", 
                                 "Latest X days of Volatility Returns for each session",
                                 "Probability Matrix",
                                 "Custom Normalised Returns",
                                 'Event Specific Distro'])


# Defining GitHub Repo
repo_name='DistributionProject'
branch='main'
plots_directory="Intraday_data_files_stats_and_plots_folder"
plot_url_base=f"https://raw.githubusercontent.com/krishangguptafibonacciresearch/{repo_name}/{branch}/{plots_directory}/"

@st.cache_data
def load_file_manifest(directory, plot_url_base):
    """Cached function to scan directory and build manifest of files"""
    plot_urls = []
    intervals = []
    instruments = []
    sessions = []
    latest_custom_days_urls = []
    
    if not os.path.exists(directory):
        st.warning(f"Directory not found: {directory}. Some features may be unavailable.")
        return plot_urls, intervals, instruments, sessions, latest_custom_days_urls

    for file in os.scandir(directory):
        if file.is_file():
            if file.name.endswith('.png'):
                plotfile_content = file.name.split('_')
                plot_url = plot_url_base + file.name
                instrument = plotfile_content[0]
                interval = plotfile_content[1]
                return_type = plotfile_content[2]

                intervals.append(interval)
                instruments.append(instrument)
                plot_urls.append({
                    "url": plot_url,
                    "instrument": instrument,
                    "interval": interval,
                    "return_type": return_type,
                    "stats_url": (plot_url_base + f'{instrument}_{interval}_{return_type}_stats.csv').replace('Volatility', 'Volatility_Returns')
                })
            elif file.name.endswith('.csv') and 'latest_custom_days' in file.name:
                if 'stats' not in str(file.name):
                    latest_custom_days_content = file.name.split('_')
                    latest_custom_days_url = plot_url_base + file.name
                    joined_session = "_".join(latest_custom_days_content[0:-7:1])
                    spaced_session = " ".join(joined_session.split('_'))
                    instrument = latest_custom_days_content[-1].replace('.csv', '')
                    interval = latest_custom_days_content[-2]
                    return_type = latest_custom_days_content[-4]

                    sessions.append(spaced_session)
                    latest_custom_days_urls.append({
                        "url": latest_custom_days_url,
                        'stats_url': plot_url_base + (file.name).split('.')[0] + '_stats.csv',
                        "instrument": instrument,
                        "interval": interval,
                        "return_type": return_type,
                        "session": [joined_session, spaced_session]
                    })
    
    return plot_urls, intervals, instruments, sessions, latest_custom_days_urls

# Load file manifest once with caching
plot_urls, intervals, instruments, sessions, latest_custom_days_urls = load_file_manifest(plots_directory, plot_url_base)

# Storing unique lists to be used later in separate drop-downs
unique_intervals = list(set(intervals)) #Interval drop-down (1hr,15min,etc)
unique_instruments=list(set(instruments)) #Instrument/ticker drop-down (ZN, ZB,etc)
unique_sessions=list(set(sessions)) #Session drop-downs (US Mid,US Open,etc)
unique_versions=['Absolute','Up','Down','No-Version']#Version drop-downs for Probability Matrix
latest_days=[14,30,60,120,240,'Custom']
data_type = ['Non-Event' , 'All data']  #type of data to use when forming the Probability Matrix


# The  default option when opening the app
desired_interval = '1h'
desired_instrument='ZN'
desired_version='Absolute'


# Set the desired values in respective drop-downs.
# Interval drop-down
if desired_interval in unique_intervals:
    default_interval_index = unique_intervals.index(desired_interval)  # Get its index
else:
    default_interval_index = 0  # Default to the first element

# Instrument drop-down
if desired_instrument in unique_instruments:
    default_instrument_index = unique_instruments.index(desired_instrument)  # Get its index
else:
    default_instrument_index = 0  # Default to the first element

# Version drop-down
if desired_version in unique_versions:
    default_version_index = unique_versions.index(desired_version)  # Get its index
else:
    default_version_index = 0 # Default to the first element


#Define tabs:
with tab1:

        # Set title
        st.title("Combined Plots for all sessions")

        # Create drop-down and display it on the left permanantly
        x= st.sidebar.selectbox("Select Interval",unique_intervals,index=default_interval_index)
        y= st.sidebar.selectbox("Select Instrument",unique_instruments,index=default_instrument_index)

        # Create checkboxes for type of return
        vol_return_bool = st.checkbox("Show Volatility Returns (bps)")
        return_bool = st.checkbox("Show Session Returns (bps)")

        
        # Store in session state
        st.session_state.x = x
        st.session_state.y = y

    
        # Get urls of the returns and volatility returns plot.
        filtered_plots = [plot for plot in plot_urls if plot["interval"] == x and plot["instrument"] == y]

        # Set volatility returns on 0th index and returns on 1st index. (False gets sorted first)
        filtered_plots = sorted(
            filtered_plots,
            key=lambda plot: (plot["return_type"] == "Returns", plot["return_type"])
        ) 

        # As per checkbox selected, modify the filtered_plots list.
    

        if vol_return_bool and return_bool:
            display_text='Displaying plots for all available Returns type.'
            return_type='Session_and_Volatility_Returns'

        elif vol_return_bool:
            display_text='Displaying plots for Volatility Returns only.'
            for index,fname in enumerate(filtered_plots):
                if 'Volatility' not in fname['return_type']:
                    filtered_plots.pop(index)
            return_type='Volatility_Returns'
        
        elif return_bool:
            display_text='Displaying plots for Session Returns only.'
            for index,fname in enumerate(filtered_plots):
                if 'Returns' not in fname['return_type']:
                    filtered_plots.pop(index)
            return_type='Session_Returns'
        
        else:
            filtered_plots=[]
            display_text=''
        st.markdown(f"<p style='color:red;'>{display_text}</p>", unsafe_allow_html=True)


        # Display plots and stats
        try:
            if filtered_plots:
                all_dataframes=[]
                tab1_sheet_names=[]
                image_url_list=[]
                tab1_image_names=[]
                for plot in filtered_plots:
                    caption = f"{plot['return_type'].replace('Returns', 'Returns Distribution').replace('Volatility', 'Volatility Distribution')}"
                    st.subheader(caption + ' Plot')
                    st.image(plot['url'],caption=caption,use_container_width=True)
                    st.subheader('Descriptive Statistics')
                    st.dataframe(
                        load_csv_from_url(plot['stats_url']),
                        use_container_width=True
                    )

                    # Save Stats dataframes into a list
                    all_dataframes.append(load_csv_from_url(plot['stats_url']))
                    tab1_sheet_names.append(caption+' Stats')

                    # Save images into a list
                    image_url_list.append(plot['url'])
                    tab1_image_names.append(f'{y}_{x}_{caption}')

                # Download Stats dataframes as Excel
                excel_file = download_combined_excel(
                    df_list=all_dataframes,
                    sheet_names=tab1_sheet_names,
                    skip_index_sheet=tab1_sheet_names
                )

                # Provide the Excel download link
                st.download_button(
                    label="Download Descriptive Statistics Data for selected Return type(s)",
                    data=excel_file,
                    file_name=f'{return_type}_{x}_{y}_stats.xlsx',
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                # Provide plots download link

                if "button_clicked" not in st.session_state:
                    st.session_state["button_clicked"] = False  # To track if the button is clicked
                    st.session_state["image_bytes_list"] = None  # To store downloaded images

                # Display the button
                if st.button("Download Image Plots"):
                    # Show the "Please wait..." message in red
                    st.session_state["button_clicked"] = True
                    wait_placeholder = st.empty()

                    # Display "Please wait..." in red
                    wait_placeholder.markdown("<span style='color: green;'>Please wait...</span>", unsafe_allow_html=True)

                    process_images(image_url_list)
                        
                    # Remove the "Please wait..." message
                    wait_placeholder.empty()
                # Handle the state when button is clicked and images are ready
                if st.session_state["image_bytes_list"] is not None:
                    st.markdown(
                        "<span style='color: white;'>(Following images are ready for download):</span>",
                        unsafe_allow_html=True
                    )
                    for img_byte, img_name in zip(st.session_state["image_bytes_list"], tab1_image_names):
                        st.download_button(
                            label=f"Download {img_name.split('_')[-1]} plot",
                            data=img_byte,
                            file_name=img_name + ".png",
                            mime="image/png"
                        )

            else:
                if vol_return_bool or return_bool:
                    st.write("No plots found for the selected interval and instrument.")
                else:
                    st.write('Please select Return type!')

        except FileNotFoundError as e:
            print(f'File not found: {e}. Please try again later.')

with tab2:
    
        st.title("Get Volatility Returns for custom days")
        
        # Use stored values from session state
        x = st.session_state.get("x", list(unique_intervals)[0])
        y = st.session_state.get("y", list(unique_instruments)[0])
        
        # Show the session dropdown
        z = st.selectbox("Select Session", unique_sessions)

        # Select number of days to analyse
        get_days=st.selectbox("Select number of days to analyse", latest_days,index=0)
        get_days_val=get_days

        if get_days=='Custom':
            enter_days=st.number_input(label="Enter the number of days:",min_value=1, step=1)
            get_days_val=enter_days

            
        filtered_latest_custom_days_csvs = [data for data in latest_custom_days_urls if data["interval"] == x  and data["instrument"] ==y and data['session'][1]==z]
        try:
            if filtered_latest_custom_days_csvs:
                for latest_custom_day_csv in filtered_latest_custom_days_csvs:
                    st.subheader(f"Volatility Returns for Latest {get_days_val} day(s) of the session: {(latest_custom_day_csv['session'])[1]}")
        
                    df=(load_csv_from_url(latest_custom_day_csv['url']))
                    latest_custom_data_csv=get_volatility_returns_csv_custom_days(target_csv=df.iloc[-1*get_days_val:],
                                                                                target_column=df.columns[1]
                    )
                    latest_custom_data_csv.reset_index(inplace=True,drop=True)
                    st.dataframe(latest_custom_data_csv,use_container_width=True)

                    st.subheader("Descriptive Statistics")
                    whole_data_stats_csv=(load_csv_from_url(latest_custom_day_csv['stats_url'])) #originally generated

                    latest_custom_data_stats_csv=get_volatility_returns_csv_stats_custom_days(target_csv=latest_custom_data_csv,
                                                                    target_column=latest_custom_data_csv.columns[1])

                    st.dataframe(latest_custom_data_stats_csv,use_container_width=True)

                
                    # Combine the DataFrames into an Excel file
                    excel_file = download_combined_excel(
                        df_list=[latest_custom_data_csv, latest_custom_data_stats_csv],
                        sheet_names=['Volatility Returns', 'Descriptive Statistics'],
                        skip_index_sheet=['Volatility Returns'],
                    )

                    # Provide the download link
                    st.download_button(
                        label="Download Returns and Statistical Data",
                        data=excel_file,
                        file_name=f'{z}_latest_{get_days_val}_Volatility_Returns_{x}_{y}.xlsx',
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
            else:
                st.write("No data found for the selected session.")
        except FileNotFoundError as e:
            print(f'File not found: {e}. Please try again later.')
        

with tab3:
        try:
            st.title("Probability Matrix (Unconditional)")
            # Use stored values from session state
            x = st.session_state.get("x", list(unique_intervals)[0])
            y = st.session_state.get("y", list(unique_instruments)[0])
            if 'h' in x:
                # Show the version dropdown
                version_value = st.selectbox("Select Version",unique_versions,index=default_version_index)

                data_type = st.selectbox("Select type of data to use", data_type , index=default_version_index)

                # Select bps to analyse
                enter_bps=st.number_input(label="Enter the number of bps:",min_value=0.0, step=0.5)
                st.caption("Note: The value must be a float and increases in steps of 0.5. Eg 1, 1.5, 2, 2.5, etc") 
                st.caption("The probability matrix rounds offs any other bps value into this format in the output.")

                # Select number of hours to analyse
                enter_hrs=st.number_input(label="Enter the number of hours:",min_value=1, step=1)
                st.caption("Note: The value must be an integer and increase in steps of 1. Eg 1, 2, 3, 4, etc.")
            
                # Get the probability matrix
                v=version_value
                
                prob_matrix_dic=GetMatrix(enter_bps,enter_hrs,x,y, data_type , version=version_value)
                st.subheader(f"Probability of bps ({v})  > {abs(enter_bps)} bps within {enter_hrs} hrs")

                # Store > probability in a small dataframe
                prob_df=pd.DataFrame(columns=['Description','Value'],
                            data=[[f'Probability of bps ({v})  > {abs(enter_bps)} bps within {enter_hrs} hrs',
                                str(round(prob_matrix_dic[v]['>%'],2))+'%'] ]
                )
                # Store <= probability in the dataframe
                prob_df.loc[len(prob_df)] = [f'Probability of bps ({v})  <= {abs(enter_bps)} bps within {enter_hrs} hrs',
                                            str(round(prob_matrix_dic[v]['<=%'],2))+'%']
                
                # Display the probability dataframe
                st.dataframe(prob_df,use_container_width=True)

                # Display the probability plots
                st.subheader(f"Probability Plot for {enter_bps} bps ({v}) movement in {enter_hrs} hrs")
                st.pyplot(prob_matrix_dic[v]['Plot'])

                st.subheader("Probability Plot for max(high-open , open-low)")
                st.pyplot(prob_matrix_dic['OH_OL_plot']['Plot'])

                # Display the probability matrix
                my_matrix=prob_matrix_dic[v]['Matrix']
                my_matrix.columns=[str(i)+' hr' for i in my_matrix.columns]
                my_matrix.index=[str(i)+' bps' for i in my_matrix.index]
                st.subheader(f"Probability Matrix of Pr(bps ({v}) >)")
                st.dataframe(my_matrix)


                # Combine the DataFrames into an Excel file
                my_matrix_list=[]
                my_matrix_ver=[]
                for ver in list(prob_matrix_dic.keys()):
                    if(ver != 'OH_OL_plot'):
                        my_matrix_list.append(prob_matrix_dic[ver]['Matrix'])
                        my_matrix_ver.append(f'{ver} bps Probability Matrix (> form)')
            
                excel_file = download_combined_excel(
                    df_list=my_matrix_list,
                    sheet_names=my_matrix_ver,
                    skip_index_sheet=[]
                )

                # Provide the download link for plots
                valid_keys = []  #first remove the OH_OL_plot key.
                for ver in prob_matrix_dic.keys():
                    if(ver != "OH_OL_plot"):
                        valid_keys.append(ver)
                st.download_button(
                    label=f"Download the Probability Matrices for version(s): bps {", bps ".join(list(valid_keys))}",
                    data=excel_file,
                    file_name=f"Probability Matrix_{'_'.join(my_matrix_ver)}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
                # Provide plots download link
                if "tab3_button_clicked" not in st.session_state:
                    st.session_state["tab3_plots_button_clicked"] = False  # To track if the button is clicked
                    st.session_state["tab3_plots_ready"] = None 

                # Display the button
                if st.button("Download Image Plots",key='tab3_button'):
                    # Show the "Please wait..." message in red
                    st.session_state["tab3_plots_button_clicked"] = True
                    wait_placeholder2 = st.empty()

                    # Display "Please wait..." in red
                    wait_placeholder2.markdown("<span style='color: green;'>Please wait...</span>", unsafe_allow_html=True)
            
                    
                    # Handle the state when button is clicked and images are ready
                    if st.session_state["tab3_plots_ready"] is not None:
                        st.markdown(
                            "<span style='color: white;'>(Following images are ready for download):</span>",
                            unsafe_allow_html=True
                        )
        
                    for ver,_ in prob_matrix_dic.items():
                        if(ver != 'OH_OL_plot'):
                            my_img_data = download_img_via_matplotlib(prob_matrix_dic[ver]['Plot'])
                            st.download_button(
                                label=f"Download the Probability Plots for version: bps {ver}",
                                data=my_img_data,
                                file_name=f"Probability Matrix_{ver}.png",
                                mime="image/png"
                            )
                    
                    # Remove the "Please wait..." message
                    wait_placeholder2.empty()
                
            else:
                st.write("Please select 1h interval.")
        except:
            display_text='1h interval data unavailable for the current ticker.'
            st.markdown(f"<p style='color:red;'>{display_text}</p>", unsafe_allow_html=True)

with tab4:
            try:
                # Protected tab
                # Add password
                PASSWORD = "distro" 

                # Initialize authentication state
                if "authenticated" not in st.session_state:
                    st.session_state.authenticated = False

                if not st.session_state.authenticated:
                    st.header("This tab is Password Protected🔒")
                    password = st.text_input("Enter Password:", type="password")
                    
                    if st.button("Login"):
                        if password == PASSWORD:
                            st.session_state.authenticated = True
                            st.rerun()
                        else:
                            st.error("Incorrect password. Try again.")
                else:
                    st.header("Authorised ✅")
                    st.write("This tab contains sensitive information.")
                    
                    if st.button("Logout"):
                        st.session_state.authenticated = False
                        st.rerun()
                    

                if st.session_state.authenticated==True:
                    # Use stored values from session state
                    x = st.session_state.get("x", list(unique_intervals)[0])
                    y = st.session_state.get("y", list(unique_instruments)[0])

                    st.title("Custom Filtering")

                    # Default sessions:
                    
                    # Show the version dropdown
                    version_value = st.selectbox("Select Version",unique_versions.copy(),index=default_version_index,
                                                key='tab4_v')

                    # Select bps to analyse
                    enter_bps=st.number_input(label="Enter the Observed movement in bps:",min_value=0.00,key='tab4_bps')

                    # Select Multiple Sessions

                    # Add custom session via button
                    default_text=f'Distribution of bps ({version_value}) Returns {y} with returns calculated for every {x}'
                    finalname=default_text
                    final_list=[]
                    
                    filter_sessions=False
                    
                    # Not include intervals
                    if 'd' not in x:
                        st.subheader('Add Custom Session')
                        tab4check=st.checkbox(label='Add Custom Session',key='tab4check')

                        if tab4check:
                            # Add Checkbox to filter by starting day
                            tab4check1=st.checkbox(label='Calculate Custom Time Difference',key='tab4check1')
                            if tab4check1:
                                # Date inputs
                                start_date = st.date_input(label="Start Date (YYYY/MM/DD)", value=datetime.today().date())
                                end_date = st.date_input(label="End Date (YYYY/MM/DD)", value=datetime.today().date())

                                # Time inputs
                                start_time = st.time_input(label="Start Time (HH:MM)",value='now',help='Directly Type Time in HH:MM')
                                end_time = st.time_input(label="End Time (HH:MM)",value='now',help='Directly Type Time in HH:MM')
                            
                                # Combine date and time into datetime objects
                                start_datetime = datetime.combine(start_date, start_time)
                                end_datetime = datetime.combine(end_date, end_time)

                                # Calculate time difference
                                time_diff = end_datetime - start_datetime

                                # Extract hours and minutes
                                hours, remainder = divmod(time_diff.total_seconds(), 3600)
                                minutes = remainder / 60

                                display_text1=(f"Time Difference: {int(hours)} hours and {int(minutes)} minutes")
                                display_text2=(f"Approx Difference (Hrs): {round(hours+minutes/60,1)} hours")
                                display_text3=(f"Approx Difference (Mins): {int(hours*60+minutes)} minutes")
                                st.markdown(f"<p style='color:red; font-size:14px;'>{display_text1}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='color:red; font-size:14px;'>{display_text2}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='color:red; font-size:14px;'>{display_text3}</p>", unsafe_allow_html=True)

                            # 1. Select Start time in ET
                            enter_start=st.number_input(label="Enter the start time in ET",min_value=0, max_value=23, step=1)
                            st.caption("Note: The value must be an integer and increase in steps of 1. Eg 1, 2, 3, 4, etc.")
                            

                            # 2. Select number of hours to analyse post the start time
                            enter_hrs=st.number_input(label=f"Enter the time (multiple of {x}) to be searched post the selected time",min_value=0, step=1)
                            st.caption("Note: The value must be an integral multiple of the interval selected")


                            # Add Checkbox to filter by starting day
                            tab4check2=st.checkbox(label='Filter by Starting Day',key='tab4check2')
                            day_list=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
                            
                            # Add Selectbox to select the starting day
                            if tab4check2==True:
                                enter_start_day=st.selectbox("Select Starting Day",day_list,index=0,
                                                key='tab4_sd')
                            else:
                                enter_start_day=""


                        
                        # Combine default and custom time filters. filter_sessions1=default, filter_sessions2=custom
                            filter_sessions1=[]
                            filter_sessions2=[]
                            filter_sessions1.append((enter_start,enter_hrs,enter_start_day))
                    
                        # Combine the two
                            filter_sessions=list(set(filter_sessions1+filter_sessions2))


                    # Give the name to include ticker,interval,time,day,start_date and end_date.
                    if filter_sessions==False:
                        filename=default_text
                    else:
                        mysession=f'{filter_sessions[0][2]} {filter_sessions[0][0]} ET to {filter_sessions[0][0]} ET+{filter_sessions[0][1]}{x[-1]}'
                        finalname=f'{default_text} for session:{mysession}'

                    # Select the dataframe for Hour interval
                    selected_df=custom_filtering_dataframe.get_dataframe(x,y,'Intraday_data_files_pq')

                    # Extract start and end dates
                    finalcsv=selected_df.copy()
                    finalcsv.index=finalcsv[finalcsv.columns[-1]]
                    finalcsv.drop_duplicates(inplace=True)
                    finalcsv.dropna(inplace=True,how='all') 
                    finalcsv.sort_index(inplace=True)
                    finalcsv = finalcsv.loc[~finalcsv.index.duplicated(keep='last')]
                    finalstart=str(finalcsv.index.to_list()[0])[:10]
                    finalend=str(finalcsv.index.to_list()[-1])[:10]


                    if filter_sessions:
                        # Filter the dataframe as per selections
                        filtered_df=custom_filtering_dataframe.filter_dataframe(selected_df,
                                                                                filter_sessions,
                                                                                day_dict="",#time_day_dict,
                                                                                timezone_column='US/Eastern Timezone',
                                                                                target_timezone='US/Eastern',
                                                                                interval=x,
                                                                                ticker=y)
                        finalname+=f' for dates:{finalstart} to {finalend}'
                        # Stats and Plots
                        stats_plots_dict=custom_filtering_dataframe.calculate_stats_and_plots(filtered_df,
                                                                            finalname,
                                                                            version=version_value,
                                                                            check_movement=enter_bps,
                                                                            interval=x,
                                                                            ticker=y,
                                                                            target_column='Group')

                    else:
                        finalname=f'{default_text} for dates:{finalstart} to {finalend}'
                        filtered_df=custom_filtering_dataframe.filter_dataframe(selected_df,
                                                                                "",
                                                                                "",
                                                                                'US/Eastern Timezone',
                                                                                'US/Eastern',
                                                                                x,
                                                                                y)
                        # Stats and Plots
                        stats_plots_dict=custom_filtering_dataframe.calculate_stats_and_plots(filtered_df,
                                                                            finalname,
                                                                            version=version_value,
                                                                            check_movement=enter_bps,
                                                                            interval=x,
                                                                            ticker=y,
                                                                            target_column='US/Eastern Timezone')

        
                    
                    # Add Widgets:
                    # Dataframe
                    st.subheader('Filtered Dataframe')
                    st.text(f'Ticker: {y}')
                    st.text(f'Interval: {x}')
                    st.text(f'Dates: {finalstart} to {finalend}')
                    # if filter_sessions==False:
                    #     session_text="None"
                    # else:
                    #     session_text=f'Start Time:{filter_sessions[0]}, Start Day:{filter_sessions[2]}, Filter for next {filter_sessions[1]} units post '
                    # st.text(f'Filters Applied: {session_text}')
                    st.dataframe(filtered_df,use_container_width=True)


                    # Display the  stats dataframe
                    stats_df=stats_plots_dict['stats']
                    st.dataframe(stats_df,use_container_width=True)

                    # Store > probability in a small dataframe
                    prob_df=pd.DataFrame(columns=['Description','Value'],
                                data=[[f'Probability of bps ({version_value})  > {abs(enter_bps)}',
                                    str(round(stats_plots_dict['%>'],2))+'%'] ]
                    )
                    # Store <= ZScore
                    prob_df.loc[len(prob_df)] =[f'Probability of bps ({version_value})  <= {abs(enter_bps)}',
                                    str(round(stats_plots_dict['%<='],2))+'%']
                    
                    prob_df.loc[len(prob_df)] =[f'ZScore for ({version_value}) bps <=  {enter_bps} bps',
                                    str((stats_plots_dict['zscore<=']))]
                

                    # Display the probability dataframe
                    st.dataframe(prob_df,use_container_width=True)


                    # Display the probability plot
                    st.subheader(f"Probability Plot for {enter_bps} bps ({version_value}) movement")
                    st.pyplot(stats_plots_dict['plot'])
                

                    # Combine the DataFrames into an Excel file (Convert datetime values to text)
                    filtered_df[filtered_df.columns[-3]]=filtered_df[filtered_df.columns[-3]].astype(str) # Datetime column
                    my_matrix_list=[filtered_df,
                                    prob_df,
                                    stats_df]
                    my_matrix_ver=[f'{x}_{y}_{finalstart} to {finalend}','Probability','Descriptive Statistics']
                
                    excel_file = download_combined_excel(
                        df_list=my_matrix_list,
                        sheet_names=my_matrix_ver,
                        skip_index_sheet=[]
                    )

                    # Provide the download link for plots
                    st.download_button(
                        label="Download Excels",
                        data=excel_file,
                        file_name=f"Probability_Stats_Excel_{finalname}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    my_img_data= download_img_via_matplotlib(stats_plots_dict['plot'])
                    st.download_button(
                            label=f"Download the Probability Plots",
                            data=my_img_data,
                            file_name=f"Probability Plot.png",
                            mime="image/png"
                        )
            except UnboundLocalError as uble:
                display_text=f'{y} Data unavailable for {x} interval.'
                st.markdown(f"<p style='color:red;'>{display_text}</p>", unsafe_allow_html=True)

            except Exception as e:
                display_text='Some error occured. Please try some other parameters and re-run.'
                st.text(e)
                st.markdown(f"<p style='color:red;'>{display_text}</p>", unsafe_allow_html=True)

with tab5:
    st.title('Event Specific Distro')
    
    # Clean up any lingering matplotlib figures from other tabs
    plt.close('all')
    gc.collect()
    
    # Define a cached function to load event data with memory optimization
    @st.cache_data
    def load_event_data():
        """Load event timestamps and OHLC data with memory-efficient techniques"""
        all_event_ts = None
        ohcl_1h = None
        
        # Load event timestamps - only the columns we need with optimized dtypes
        for file in os.scandir("Intraday_data_files_processed_folder_pq"):
            if file.name == "ZN_1h_events_tagged_target_tz.parquet":
                # Load with specific dtypes to reduce memory
                all_event_ts = pd.read_parquet(
                    file.path, 
                    engine='pyarrow',
                    columns=['timestamp', 'events']  # Only load needed columns
                )
                # Convert events to categorical to save memory
                if 'events' in all_event_ts.columns:
                    all_event_ts['events'] = all_event_ts['events'].astype('category')
                break
        
        if all_event_ts is not None:
            all_event_ts['timestamp'] = pd.to_datetime(all_event_ts.timestamp, errors='coerce').dt.tz_localize('US/Eastern')
            # Force garbage collection after timestamp conversion
            gc.collect()
        
        # Load OHLC data - only the columns we need with float32 precision
        pattern = re.compile(r"Intraday_data_ZN_1h_2022-12-20_to_(\d{4}-\d{2}-\d{2})\.parquet")
        for file in os.scandir('Intraday_data_files_pq'):
            if file.is_file():
                match = pattern.match(file.name)
                if match:
                    # Load with float32 instead of float64 to halve memory usage
                    ohcl_1h = pd.read_parquet(
                        os.path.join("Intraday_data_files_pq", file.name), 
                        engine='pyarrow',
                        columns=['Open', 'High', 'Low', 'Close']  # Only load OHLC columns
                    )
                    # Convert to float32
                    for col in ['Open', 'High', 'Low', 'Close']:
                        if col in ohcl_1h.columns:
                            ohcl_1h[col] = ohcl_1h[col].astype('float32')
                    break
        
        if ohcl_1h is not None:
            ohcl_1h['US/Eastern Timezone'] = pd.to_datetime(ohcl_1h.index, errors='coerce', utc=True)
            ohcl_1h['US/Eastern Timezone'] = ohcl_1h['US/Eastern Timezone'].dt.tz_convert('US/Eastern')
            # Force garbage collection after timezone conversion
            gc.collect()
        
        return all_event_ts, ohcl_1h
    
    # Load data once with progress indicator
    with st.spinner('Loading event data... This may take a moment on first load.'):
        all_event_ts, ohcl_1h = load_event_data()
    
    if all_event_ts is None or ohcl_1h is None:
        st.error("Required data files not found. Please ensure the necessary parquet files are in the correct directories.")
    else:
        events = ['CPI', 'PPI', 'PCE Price Index', 'Non Farm Payrolls', 'ISM Manufacturing PMI', 'ISM Services PMI',
                  'S&P Global Manufacturing PMI', 'S&P Global Services PMI', 'Michigan',
                  'Jobless Claims' , 'ADP' , 'JOLTs' , 'Challenger Job Cuts' , 'Fed Interest Rate Decision' , 
                  'GDP Price Index QoQ Adv' , 'Retail Sales' , 'Fed Press Conference', 'FOMC Minutes']
        
        selected_event = st.selectbox("Select an event:" , events)
        duration = ['pre event (8 hr before event)' , 'immediate reaction (1 hr after the event)']
        dur = st.selectbox("Select duration: " , duration)

        ############## Added by Yaman ########################################################################
        filter_isolated = st.checkbox(
        "Exclude events with any other announcement ±2 hours",
        help="Only show events that have no other events in the surrounding time window."
        )
        ######################################################################################################

        my_dict = {"pre event (8 hr before event)": 1 , "immediate reaction (1 hr after the event)": 2}

        ############################################### Changed a bit by Yaman ##########################################################################
        custom = st.checkbox('Custom time')
        if(custom):
            delta = st.number_input("Enter the number of hours:", min_value=-1000, max_value=1000 , value=0, step=1)
            final_df = calc_event_spec_returns(selected_event, all_event_ts, ohcl_1h , 3 , events, delta, filter_isolated , 2)
            # Check memory before plotting
            check_memory_and_cleanup()
            plot_event_spec_returns(final_df , selected_event , dur)
        else:
            final_df = calc_event_spec_returns(selected_event, all_event_ts, ohcl_1h , my_dict[dur], events, 0 , filter_isolated , 2)
            # Check memory before plotting
            check_memory_and_cleanup()
            plot_event_spec_returns(final_df , selected_event , dur)

##########################################################