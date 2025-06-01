import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION ---
st.set_page_config(page_title="Peppered Moth Evolution", layout="wide", initial_sidebar_state="expanded")

# --- HEADER ---
st.title("Peppered Moth Evolution Simulator")
st.markdown("""
This app simulates the evolution of **Peppered Moths** (*Biston betularia*) during the Industrial Revolution.
It demonstrates how environmental change (pollution) affects natural selection and camouflage effectiveness.
""")

# Add educational context
with st.expander("About This Simulation"):
    st.markdown("""
    **Historical Context**: During the Industrial Revolution, tree trunks became darker due to soot and pollution. 
    This environmental change favored dark-colored moths over light-colored ones, as they were better camouflaged 
    against the darkened bark. This is a classic example of natural selection in action.
    
    **How it works**: 
    - Moths with better camouflage have higher survival rates
    - Survivors reproduce and pass on their traits
    - Mutations occasionally change moth color
    - Population frequencies shift over generations
    """)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("🔧 Simulation Parameters")

# Population settings
st.sidebar.subheader("Population Settings")
initial_population = st.sidebar.slider("Initial Population", 100, 1000, 300, step=50, 
                                     help="Total number of moths at the start")
generations = st.sidebar.slider("Number of Generations", 5, 100, 30,
                               help="How many generations to simulate")

# Environmental settings
st.sidebar.subheader("Environmental Factors")
pollution_start = st.sidebar.slider("Initial Pollution Level", 0.0, 1.0, 0.3, step=0.05,
                                   help="0 = Clean environment, 1 = Heavily polluted")
pollution_rate = st.sidebar.slider("Pollution Change Rate per Generation", -0.05, 0.05, 0.01, step=0.01,
                                  help="How pollution changes each generation")

# Genetic settings
st.sidebar.subheader("Genetic Factors")
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 0.1, 0.01, step=0.005,
                                 help="Probability of color mutation per offspring")

# Animation settings
st.sidebar.subheader("Animation Settings")
speed = st.sidebar.slider("Animation Speed (seconds)", 0.01, 1.0, 0.2, step=0.01,
                         help="Delay between generations")

# Control buttons in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Controls")
col1, col2 = st.sidebar.columns(2)
run_button = col1.button("Run Simulation", type="primary", use_container_width=True)
reset_button = col2.button("Reset", use_container_width=True)

# --- MAIN CONTENT AREA ---
if reset_button:
    st.rerun()

if run_button:
    # Initialize population
    white_pop = int(initial_population * 0.5)
    dark_pop = initial_population - white_pop
    pollution_level = pollution_start

    # Data storage
    white_history = []
    dark_history = []
    pollution_history = []
    generation_data = []

    # Create layout containers
    status_container = st.container()
    main_container = st.container()
    
    with status_container:
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        gen_metric = status_col1.empty()
        white_metric = status_col2.empty()
        dark_metric = status_col3.empty()
        pollution_metric = status_col4.empty()

    with main_container:
        # Create three columns for better layout
        viz_col, graph_col, info_col = st.columns([2, 2, 1])
        
        with viz_col:
            st.subheader("Environment & Camouflage")
            camouflage_placeholder = st.empty()
            
        with graph_col:
            st.subheader("Population Trends")
            graph_placeholder = st.empty()
            
        with info_col:
            st.subheader("Statistics")
            stats_placeholder = st.empty()

    # Progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    # Run simulation
    for gen in range(generations):
        # Update progress
        progress = (gen + 1) / generations
        progress_bar.progress(progress)
        progress_text.text(f"Running simulation... Generation {gen + 1}/{generations}")
        
        # --- Survival Probabilities ---
        white_survival = max(0.05, 1.0 - pollution_level)
        dark_survival = max(0.05, pollution_level)

        white_survivors = np.random.binomial(white_pop, white_survival)
        dark_survivors = np.random.binomial(dark_pop, dark_survival)

        # --- Reproduction and Mutation ---
        white_offspring = white_survivors * 2
        dark_offspring = dark_survivors * 2

        to_dark = np.random.binomial(white_offspring, mutation_rate)
        to_white = np.random.binomial(dark_offspring, mutation_rate)

        white_pop = white_offspring - to_dark + to_white
        dark_pop = dark_offspring - to_white + to_dark

        # --- Store History ---
        white_history.append(white_pop)
        dark_history.append(dark_pop)
        pollution_history.append(pollution_level)
        
        total_pop = white_pop + dark_pop
        white_percentage = (white_pop / total_pop * 100) if total_pop > 0 else 0
        dark_percentage = (dark_pop / total_pop * 100) if total_pop > 0 else 0
        
        generation_data.append({
            'generation': gen + 1,
            'white_pop': white_pop,
            'dark_pop': dark_pop,
            'total_pop': total_pop,
            'pollution': pollution_level,
            'white_pct': white_percentage,
            'dark_pct': dark_percentage
        })

        # --- Update Status Metrics ---
        gen_metric.metric("Generation", f"{gen + 1}/{generations}")
        white_metric.metric("White Moths", white_pop, f"{white_percentage:.1f}%")
        dark_metric.metric("Dark Moths", dark_pop, f"{dark_percentage:.1f}%")
        pollution_metric.metric("Pollution Level", f"{pollution_level:.2f}", f"{pollution_rate:+.3f}")

        # --- Visualizing Environment (Camouflage) ---
        env_gray = pollution_level
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.set_facecolor((env_gray, env_gray, env_gray))
        ax2.set_title(f"Tree Bark Environment (Pollution: {pollution_level:.2f})", fontsize=12, pad=20)
        ax2.axis('off')

        # Display representative sample of moths
        total_display = min(white_pop + dark_pop, 150)
        if total_display > 0:
            white_ratio = white_pop / (white_pop + dark_pop)
            n_white = int(total_display * white_ratio)
            n_dark = total_display - n_white

            if n_white > 0:
                x_white, y_white = np.random.rand(n_white), np.random.rand(n_white)
                ax2.scatter(x_white, y_white, color="white", edgecolor='black', s=40, 
                           alpha=0.8, label=f"White Moths ({n_white})")
            
            if n_dark > 0:
                x_dark, y_dark = np.random.rand(n_dark), np.random.rand(n_dark)
                ax2.scatter(x_dark, y_dark, color="black", edgecolor='white', s=40, 
                           alpha=0.8, label=f"Dark Moths ({n_dark})")
            
            ax2.legend(loc="upper right", framealpha=0.9)
        
        camouflage_placeholder.pyplot(fig2)
        plt.close(fig2)

        # --- Population Trends Graph ---
        fig1, (ax1, ax3) = plt.subplots(2, 1, figsize=(6, 6))
        
        # Population over time
        generations_x = list(range(1, len(white_history) + 1))
        ax1.plot(generations_x, white_history, label="White Moths", color="lightgray", linewidth=2, marker='o', markersize=3)
        ax1.plot(generations_x, dark_history, label="Dark Moths", color="black", linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Population")
        ax1.set_title("Population Over Time")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pollution level over time
        ax3.plot(generations_x, pollution_history, label="Pollution Level", color="brown", linewidth=2)
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Pollution Level")
        ax3.set_title("Environmental Change")
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        graph_placeholder.pyplot(fig1)
        plt.close(fig1)

        # --- Statistics Panel ---
        with stats_placeholder.container():
            st.metric("Total Population", total_pop)
            st.metric("Dominant Variant", 
                     "Dark" if dark_pop > white_pop else "White",
                     f"{max(white_percentage, dark_percentage):.1f}%")
            
            # Survival rates
            st.markdown("**Survival Rates:**")
            st.markdown(f"• White: {white_survival:.1%}")
            st.markdown(f"• Dark: {dark_survival:.1%}")
            
            # Environmental pressure
            if pollution_level < 0.3:
                pressure = "Favors Light"
            elif pollution_level > 0.7:
                pressure = "Favors Dark"
            else:
                pressure = "Neutral"
            
            st.markdown(f"**Environment:** {pressure}")

        # Update pollution level for next generation
        pollution_level = min(1.0, max(0.0, pollution_level + pollution_rate))
        
        # Animation delay
        time.sleep(speed)

    # --- Final Results ---
    progress_text.text("Simulation Complete!")
    
    # Summary statistics
    st.markdown("---")
    st.subheader("Simulation Summary")
    
    final_col1, final_col2, final_col3 = st.columns(3)
    
    with final_col1:
        st.metric("Final White Population", white_history[-1])
        st.metric("Final Dark Population", dark_history[-1])
    
    with final_col2:
        initial_white_pct = 50.0
        final_white_pct = (white_history[-1] / (white_history[-1] + dark_history[-1]) * 100) if (white_history[-1] + dark_history[-1]) > 0 else 0
        change_white = final_white_pct - initial_white_pct
        
        st.metric("White Moth %", f"{final_white_pct:.1f}%", f"{change_white:+.1f}%")
        st.metric("Dark Moth %", f"{100-final_white_pct:.1f}%", f"{-change_white:+.1f}%")
    
    with final_col3:
        st.metric("Final Pollution", f"{pollution_history[-1]:.2f}")
        st.metric("Total Generations", generations)

    # Educational conclusion
    st.markdown("---")
    st.subheader("Scientific Insights")
    
    if final_white_pct < 25:
        conclusion = "The dark moths became dominant due to increased pollution, demonstrating natural selection."
    elif final_white_pct > 75:
        conclusion = "The white moths remained dominant in the cleaner environment."
    else:
        conclusion = "Both variants coexisted, showing balanced selection pressure."
    
    st.info(f"**Conclusion:** {conclusion}")

else:
    # Show instructions when simulation hasn't been run
    st.info("Configure your simulation parameters in the sidebar and click 'Run Simulation' to begin!")
    
    # Show sample visualization
    st.subheader("What You'll See")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Environment Visualization**
        - Background color represents pollution level
        - Moths are displayed as dots on tree bark
        - Better camouflaged moths survive more often
        """)
    
    with col2:
        st.markdown("""
        **Population Tracking**
        - Real-time population graphs
        - Environmental change over time
        - Statistical summaries and metrics
        """)