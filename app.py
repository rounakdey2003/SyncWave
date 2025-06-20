import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import cv2
from camera_utils import MotionDetector

class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'

        self.data = np.load(dataset)
        self.Fs = 250
        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T
        self.events_position = self.data['epos'].T
        self.events_duration = self.data['edur'].T
        self.artifacts = self.data['artifacts'].T

        self.mi_types = {769: 'left', 770: 'right',
                        771: 'foot', 772: 'tongue', 783: 'unknown'}

    def get_trials_from_channel(self, channel=7):
        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        trials = []
        classes = []

        for index in idxs:
            try:
                type_e = self.events_type[0, index+1]
                class_e = self.mi_types[type_e]
                classes.append(class_e)

                start = self.events_position[0, index]
                stop = start + self.events_duration[0, index]
                trial = self.raw[channel, start:stop]
                trial = trial.reshape((1, -1))
                trials.append(trial)

            except:
                continue

        return trials, classes

    def get_trials_from_channels(self, channels=[7, 9, 11]):
        trials_c = []
        classes_c = []
        for c in channels:
            t, c = self.get_trials_from_channel(channel=c)
            tt = np.concatenate(t, axis=0)
            trials_c.append(tt)
            classes_c.append(c)

        return trials_c, classes_c


st.set_page_config(page_title="BCI", layout="wide",initial_sidebar_state='expanded')


st.markdown("<h1 style='text-align: center;'>Brain Computer Interection (BCI)</h1>", unsafe_allow_html=True)

st.sidebar.title("Control Panel")
subject = st.sidebar.selectbox(
    "Choose test subject",
    [f"A0{i}T" for i in range(1, 10)]
)

with open(f"{subject}.npz", "rb") as file:
    btn = st.sidebar.download_button(
        label="Download",
        data=file,
        file_name=f"{subject}.npz",
        mime="application/x-npz",
        type="primary"
    )


st.sidebar.divider()
st.sidebar.page_link(page="https://github.com/rounakdey2003/SyncWave", label=":blue-background[:blue[Github]]",
                     help='Teleport to Github',
                     use_container_width=False)

with st.container(border=True):
    st.markdown("""
    ### Note
    - LEFT hand     (C4)
    - RIGHT hand    (C3)
    - FOOT & TONGUE (Cz)

    """)


dataset = MotorImageryDataset(subject)
trials, classes = dataset.get_trials_from_channels([7, 9, 11])

tab1, tab2, tab3 = st.tabs(["Brain Activity Map", "Brain Signals Explorer", "Movement Detection"])

with tab1:
    with st.container(border=True):
        st.markdown("""
        ### Brain Activity
        - :red[**Red colors**]: Brain is very active
        - :blue[**Blue colors**]: Brain is resting

        - **Brain Part**:
            - **C3**: Left side of the brain (helps control right side of body)
            - **Cz**: Middle of the brain (helps with foot and tongue movements)
            - **C4**: Right side of the brain (helps control left side of body)
        """)
    with st.container(border=True):
        fig = make_subplots(rows=3, cols=1, 
                            subplot_titles=('Left Brain (C3)', 'Middle Brain (Cz)', 'Right Brain (C4)'),
                            vertical_spacing=0.15)

        for i, (trial, title) in enumerate(zip(trials, ['C3', 'Cz', 'C4'])):
            fig.add_trace(
                go.Heatmap(
                    z=trial,
                    showscale=(i == 0),
                    colorscale='RdBu',
                    name=title
                ),
                row=i+1, col=1
            )

        fig.update_layout(
            height=900,
            width=800,
            title_text=f"Activity Monitor",
            showlegend=False,
            font=dict(size=14)
        )

        st.plotly_chart(fig, use_container_width=True, key="brain_activity_map")

with tab2:
    with st.container(border=True):
        st.markdown("""
        ### Brain Signals
        - Each bump in the line means brain sending a tiny electrical signal
        - When the line goes up and down a lot, means brain is very active
        - When the line is flatter, means brain is more relaxed
        """)
    with st.container(border=True):
        selected_channel = st.selectbox(
            "Choose brain part",
            ["Left Brain (C3)", "Middle Brain (Cz)", "Right Brain (C4)"]
        )
        
        channel_idx = {'Left Brain (C3)': 0, 'Middle Brain (Cz)': 1, 'Right Brain (C4)': 2}
        

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            y=trials[channel_idx[selected_channel]][0],
            mode='lines',
            name='Brain Signal',
            line=dict(color='#2E86C1', width=2)
        ))
        
        fig2.update_layout(
            title=f"Brain Waves from {selected_channel}",
            yaxis_title="Signal Strength",
            xaxis_title="Time",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig2, use_container_width=True, key="brain_signal_plot")
        
with tab3:
    with st.container(border=True):
        st.markdown("""
        ### Camera-Based Movement Detection
        This feature uses your webcam to detect body movements and shows which parts of your brain would be active during these movements.
        
        - **Left Hand Movement**: Activates the right side of your brain (C4)
        - **Right Hand Movement**: Activates the left side of your brain (C3)
        - **Head Movement**: Activates the middle of your brain (Cz)
        """)
    


        example_fig = go.Figure()
        
        for i, region in enumerate(['C3', 'Cz', 'C4']):
            is_active = i % 2 == 0
            color = 'rgba(255, 0, 0, 0.7)' if is_active else 'rgba(0, 0, 255, 0.7)'
            example_fig.add_trace(go.Scatter(
                x=[i-0.4, i+0.4, i+0.4, i-0.4, i-0.4],
                y=[-0.4, -0.4, 0.4, 0.4, -0.4],
                fill="toself",
                fillcolor=color,
                line=dict(color='black'),
                mode='lines',
                name=region,
                text=f"{region}: {'Active' if is_active else 'Resting'}",
                hoverinfo='text'
            ))
            
            example_fig.add_annotation(
                x=i, y=0,
                text=region,
                showarrow=False,
                font=dict(color='white', size=14)
            )
        
        example_fig.update_layout(
            title="Example: Brain Activity Based on Movement",
            xaxis=dict(showticklabels=False, range=[-1, 3]),
            yaxis=dict(showticklabels=False, range=[-1, 1]),
            showlegend=False,
            height=300
        )
        
        st.plotly_chart(example_fig, use_container_width=True, key="example_brain_activity")

    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
        st.session_state.motion_detector = None
    
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.camera_running:
            if st.button("Start Camera", type="primary"):
                try:
                    st.session_state.motion_detector = MotionDetector()
                    if st.session_state.motion_detector.start_camera():
                        st.session_state.camera_running = True
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error starting camera: {str(e)}")
        else:
            if st.button("Stop Camera", type="primary"):
                if st.session_state.motion_detector:
                    st.session_state.motion_detector.stop_camera()
                st.session_state.camera_running = False
                st.experimental_rerun()
    
    if st.session_state.camera_running and st.session_state.motion_detector:
        col1, col2 = st.columns(2)
        
        with col1:
            camera_placeholder = st.empty()
        
        with col2:
            brain_activity_placeholder = st.empty()
        st.markdown("### Real-time Brain Signals")
        c3_signal_placeholder = st.empty()
        cz_signal_placeholder = st.empty()
        c4_signal_placeholder = st.empty()
        
        stop_button_placeholder = st.empty()
        
        try:
            while st.session_state.camera_running:
                frame = st.session_state.motion_detector.get_frame()
                if frame is None:
                    st.error("Failed to capture frame from camera")
                    break
                
                processed_frame, _ = st.session_state.motion_detector.detect_motion(frame)
                
                active_brain_regions = st.session_state.motion_detector.get_active_brain_regions()
                
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                camera_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                brain_fig = go.Figure()
                
                for i, region in enumerate(['C3', 'Cz', 'C4']):
                    info = active_brain_regions[region]
                    color = 'rgba(255, 0, 0, 0.7)' if info['active'] else 'rgba(0, 0, 255, 0.7)'
                    brain_fig.add_trace(go.Scatter(
                        x=[i-0.4, i+0.4, i+0.4, i-0.4, i-0.4],
                        y=[-0.4, -0.4, 0.4, 0.4, -0.4],
                        fill="toself",
                        fillcolor=color,
                        line=dict(color='black'),
                        mode='lines',
                        name=region,
                        text=f"{region}: {'Active' if info['active'] else 'Resting'}",
                        hoverinfo='text'
                    ))
                    
                    brain_fig.add_annotation(
                        x=i, y=0,
                        text=region,
                        showarrow=False,
                        font=dict(color='white', size=14)
                    )
                    
                    if info['active']:
                        brain_fig.add_annotation(
                            x=i, y=0.6,
                            text=f"Active: {info['body_part']}",
                            showarrow=False,
                            font=dict(color='black', size=12),
                            bgcolor='rgba(255, 255, 255, 0.7)'
                        )
                brain_fig.update_layout(
                    title="Brain Activity Based on Movement",
                    xaxis=dict(showticklabels=False, range=[-1, 3]),
                    yaxis=dict(showticklabels=False, range=[-1, 1]),
                    showlegend=False,
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                brain_activity_placeholder.plotly_chart(brain_fig, use_container_width=True, key="brain_activity_plot")
                
                region_idx = {'C3': 0, 'Cz': 1, 'C4': 2}
                region_titles = {'C3': 'Left Brain (C3)', 'Cz': 'Middle Brain (Cz)', 'C4': 'Right Brain (C4)'}
                
                c3_info = active_brain_regions['C3']
                c3_signal_data = trials[region_idx['C3']][0]
                c3_fig = go.Figure()
                
                c3_signal_segment = st.session_state.motion_detector.generate_real_time_signal('C3', c3_signal_data, num_points=100)
                
                c3_fig.add_trace(go.Scatter(
                    y=c3_signal_segment,
                    mode='lines',
                    name='C3',
                    line=dict(color='red' if c3_info['active'] else 'blue', width=2)
                ))
                
                c3_fig.update_layout(
                    title=region_titles['C3'],
                    yaxis_title="Signal Strength",
                    xaxis_title="Time",
                    height=150,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                cz_info = active_brain_regions['Cz']
                cz_signal_data = trials[region_idx['Cz']][0]
                cz_fig = go.Figure()
                
                cz_signal_segment = st.session_state.motion_detector.generate_real_time_signal('Cz', cz_signal_data, num_points=100)
                
                cz_fig.add_trace(go.Scatter(
                    y=cz_signal_segment,
                    mode='lines',
                    name='Cz',
                    line=dict(color='red' if cz_info['active'] else 'blue', width=2)
                ))
                
                cz_fig.update_layout(
                    title=region_titles['Cz'],
                    yaxis_title="Signal Strength",
                    xaxis_title="Time",
                    height=150,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                c4_info = active_brain_regions['C4']
                c4_signal_data = trials[region_idx['C4']][0]
                c4_fig = go.Figure()
                
                c4_signal_segment = st.session_state.motion_detector.generate_real_time_signal('C4', c4_signal_data, num_points=100)
                
                c4_fig.add_trace(go.Scatter(
                    y=c4_signal_segment,
                    mode='lines',
                    name='C4',
                    line=dict(color='red' if c4_info['active'] else 'blue', width=2)
                ))
                
                c4_fig.update_layout(
                    title=region_titles['C4'],
                    yaxis_title="Signal Strength",
                    xaxis_title="Time",
                    height=150,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                c3_signal_placeholder.plotly_chart(c3_fig, use_container_width=True, key="c3_signal_plot")
                cz_signal_placeholder.plotly_chart(cz_fig, use_container_width=True, key="cz_signal_plot")
                c4_signal_placeholder.plotly_chart(c4_fig, use_container_width=True, key="c4_signal_plot")
                
                current_time = int(time.time() * 1000)
                if stop_button_placeholder.button("Stop Camera", key=f"stop_in_loop_{current_time}"):
                    st.session_state.camera_running = False
                    if st.session_state.motion_detector:
                        st.session_state.motion_detector.stop_camera()
                    break
                
                time.sleep(0.1)
        except Exception as e:
            st.error(f"Error in camera processing: {str(e)}")
            st.session_state.camera_running = False
            if st.session_state.motion_detector:
                st.session_state.motion_detector.stop_camera()
    
    if not st.session_state.camera_running:
        st.info("Click 'Start Camera' to begin movement detection")
        
