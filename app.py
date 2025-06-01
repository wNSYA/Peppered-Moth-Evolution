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
    - Population grows but is limited by carrying capacity (logistic growth)
    - Mutations occasionally change moth color
    - Population frequencies shift over generations
    - **Note**: The first generation remains unchanged from initial population to show baseline
    """)

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Simulation Parameters")

# Population settings
st.sidebar.subheader("Population Settings")
initial_population = st.sidebar.slider("Initial Population", 100, 1000, 300, step=50, 
                                     help="Total number of moths at the start")
generations = st.sidebar.slider("Number of Generations", 5, 100, 30,
                               help="How many generations to simulate")

# Hard-coded parameters for realistic simulation
carrying_capacity = 2000  # Maximum sustainable population
reproduction_rate = 2.2   # Increased from 1.8 to allow better population growth

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
    # --- Population Initialization (Modified for 50/50 split) ---
    # Create exactly 50/50 split between white (0) and dark (1) moths
    half_pop = initial_population // 2
    remaining = initial_population % 2
    
    # Create equal numbers of each type
    white_moths = np.zeros(half_pop, dtype=int)  # 0 = white
    dark_moths = np.ones(half_pop, dtype=int)    # 1 = dark
    
    # Handle odd population sizes by adding one more moth randomly
    if remaining > 0:
        extra_moth = np.random.choice([0, 1], size=remaining)
        population = np.concatenate([white_moths, dark_moths, extra_moth])
    else:
        population = np.concatenate([white_moths, dark_moths])
    
    # Shuffle the population to randomize positions
    np.random.shuffle(population)
    
    # Display initial population composition
    initial_white = np.count_nonzero(population == 0)
    initial_dark = np.count_nonzero(population == 1)
    
    pollution_level = pollution_start
    
    # Data history
    white_history, dark_history, pollution_history, total_pop_history, generation_data = [], [], [], [], []

    # Create layout containers
    status_container = st.container()
    main_container = st.container()
    
    with status_container:
        status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
        gen_metric = status_col1.empty()
        total_metric = status_col2.empty()
        white_metric = status_col3.empty()
        dark_metric = status_col4.empty()
        pollution_metric = status_col5.empty()

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

    # Run simulation with population growth
    for gen in range(generations):
        # Update progress
        progress = (gen + 1) / generations
        progress_bar.progress(progress)
        progress_text.text(f"Running simulation... Generation {gen + 1}/{generations}")
        
        current_pop_size = len(population)
        
        # --- FIRST GENERATION: No changes, keep original population ---
        if gen == 0:
            # For the first generation, keep the population exactly as initialized
            # No selection, no reproduction, no mutation - just display the baseline
            
            # Set survival rates to 100% for display
            white_survival = 1.0
            dark_survival = 1.0
            
            # Population remains unchanged for generation 1
            # (population variable stays the same)
            
        else:
            # --- SUBSEQUENT GENERATIONS: Apply normal selection ---    
            # --- Fitness Calculation ---
            # Calculate base fitness (0 to 1)
            base_fitness = np.where(population == 0, 1.0 - pollution_level, pollution_level)
            
            # Apply gentler selection pressure using sigmoid-like curve
            fitness_advantage = 0.2  # Reduced from 0.3 to make selection less extreme
            baseline_survival = 0.85  # Increased from 0.7 to give higher base survival rate
            
            # Scale fitness to be less extreme: baseline + advantage based on environment
            fitness = baseline_survival + (base_fitness - 0.5) * fitness_advantage
            fitness = np.clip(fitness, 0.6, 0.95)  # Gentler range: 60%-95% survival
            
            # --- Selection (Survival of the Fittest) ---
            survivors_mask = np.random.rand(len(population)) < fitness
            survivors = population[survivors_mask]
            
            # Calculate survival rates for display
            initial_white_gen = np.count_nonzero(population == 0)
            initial_dark_gen = np.count_nonzero(population == 1)
            
            surviving_white = np.count_nonzero(survivors == 0)
            surviving_dark = np.count_nonzero(survivors == 1)
            
            white_survival = surviving_white / initial_white_gen if initial_white_gen > 0 else 0
            dark_survival = surviving_dark / initial_dark_gen if initial_dark_gen > 0 else 0
            
            # Prevent population extinction
            if len(survivors) == 0:
                # Recreate small population if extinction occurs
                survivors = np.random.choice([0, 1], size=max(10, initial_population // 10))
            
            # --- Reproduction with Logistic Growth ---
            # Calculate logistic growth factor
            # Growth slows as population approaches carrying capacity
            growth_factor = 1 - (current_pop_size / carrying_capacity)
            growth_factor = max(0.2, growth_factor)  # Increased minimum growth factor from 0.1 to 0.2
            
            # Effective reproduction rate decreases as population grows
            effective_reproduction_rate = reproduction_rate * growth_factor
            
            # Calculate target population size for next generation
            # Each survivor can potentially reproduce based on the effective rate
            expected_offspring = len(survivors) * effective_reproduction_rate
            
            # Add some randomness to reproduction
            offspring_variance = 0.2  # 20% variance
            actual_offspring = int(expected_offspring * (1 + np.random.normal(0, offspring_variance)))
            
            # Ensure population doesn't exceed carrying capacity
            next_pop_size = min(actual_offspring, carrying_capacity)
            next_pop_size = max(next_pop_size, len(survivors))  # At least maintain survivors
            
            # --- Create Next Generation ---
            if next_pop_size <= len(survivors):
                # Population is declining or stable
                offspring = np.random.choice(survivors, size=next_pop_size)
            else:
                # Population is growing - survivors + new offspring
                new_offspring_count = next_pop_size - len(survivors)
                new_offspring = np.random.choice(survivors, size=new_offspring_count)
                offspring = np.concatenate([survivors, new_offspring])
            
            # --- Mutation (Genetic variation) ---
            mutation_mask = np.random.rand(len(offspring)) < mutation_rate
            offspring[mutation_mask] = 1 - offspring[mutation_mask]  # Flip gene (0->1 or 1->0)
            
            # --- Next Generation ---
            population = offspring
        
        # Count phenotypes for visualization (this happens for all generations)
        white_pop = np.count_nonzero(population == 0)
        dark_pop = np.count_nonzero(population == 1)
        total_pop = len(population)
        
        # Store history
        white_history.append(white_pop)
        dark_history.append(dark_pop)
        total_pop_history.append(total_pop)
        pollution_history.append(pollution_level)
        
        white_percentage = (white_pop / total_pop * 100) if total_pop > 0 else 0
        dark_percentage = (dark_pop / total_pop * 100) if total_pop > 0 else 0
        
        generation_data.append({
            'generation': gen + 1,
            'white_pop': white_pop,
            'dark_pop': dark_pop,
            'total_pop': total_pop,
            'pollution': pollution_level,
            'white_pct': white_percentage,
            'dark_pct': dark_percentage,
            'growth_factor': growth_factor if gen > 0 else 1.0
        })

        # --- Update Status Metrics ---
        gen_metric.metric("Generation", f"{gen + 1}/{generations}")
        white_metric.metric("White Moths", white_pop, f"{white_percentage:.1f}%")
        dark_metric.metric("Dark Moths", dark_pop, f"{dark_percentage:.1f}%")
        pollution_metric.metric("Pollution Level", f"{pollution_level:.2f}", f"{pollution_rate:+.3f}")

        # --- Visualizing Environment (Camouflage) ---
        # Create realistic tree bark coloring based on pollution
        clean_bark_color = (0.85, 0.75, 0.65)  # Light brownish bark
        polluted_bark_color = (0.15, 0.12, 0.10)  # Dark sooty bark
        
        # Interpolate between clean and polluted colors
        bark_r = clean_bark_color[0] * (1 - pollution_level) + polluted_bark_color[0] * pollution_level
        bark_g = clean_bark_color[1] * (1 - pollution_level) + polluted_bark_color[1] * pollution_level
        bark_b = clean_bark_color[2] * (1 - pollution_level) + polluted_bark_color[2] * pollution_level
        
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.set_facecolor((bark_r, bark_g, bark_b))
        
        # Add some texture to simulate bark pattern
        np.random.seed(42)  # For consistent texture
        n_texture = 30
        texture_x = np.random.rand(n_texture)
        texture_y = np.random.rand(n_texture)
        texture_color = (bark_r * 0.7, bark_g * 0.7, bark_b * 0.7)
        ax2.scatter(texture_x, texture_y, c=[texture_color], s=20, alpha=0.3, marker='s')
        
        pollution_desc = "Clean" if pollution_level < 0.3 else "Moderately Polluted" if pollution_level < 0.7 else "Heavily Polluted"
        generation_status = " (Baseline - No Selection)" if gen == 0 else ""
        ax2.set_title(f"Tree Bark Environment - {pollution_desc}, Pollution: {pollution_level:.2f}{generation_status}", 
                     fontsize=12, pad=20)
        ax2.axis('off')

        # Display representative sample of moths (limit for visualization)
        total_display = min(total_pop, 200)  # Increased display limit
        if total_display > 0:
            white_ratio = white_pop / total_pop
            n_white = int(total_display * white_ratio)
            n_dark = total_display - n_white

            if n_white > 0:
                x_white, y_white = np.random.rand(n_white), np.random.rand(n_white)
                ax2.scatter(x_white, y_white, color="white", edgecolor='black', s=40, 
                           alpha=0.8, label=f"White Moths ({white_pop})")
            
            if n_dark > 0:
                x_dark, y_dark = np.random.rand(n_dark), np.random.rand(n_dark)
                ax2.scatter(x_dark, y_dark, color="black", edgecolor='white', s=40, 
                           alpha=0.8, label=f"Dark Moths ({dark_pop})")
            
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
        
        # Add vertical line to show when selection starts
        if len(generations_x) > 1:
            ax1.axvline(x=1.5, color='red', linestyle='--', alpha=0.5, label='Selection Starts')
            ax1.legend()
        
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
            st.metric("Total Population", total_pop, f"{total_pop - current_pop_size:+d}")
            st.metric("Dominant Variant", 
                     "Dark" if dark_pop > white_pop else "White",
                     f"{max(white_percentage, dark_percentage):.1f}%")
            
            # Survival rates
            st.markdown("**Survival Rates:**")
            if gen == 0:
                st.markdown("• White: 100% (No Selection)")
                st.markdown("• Dark: 100% (No Selection)")
            else:
                st.markdown(f"• White: {white_survival:.1%}")
                st.markdown(f"• Dark: {dark_survival:.1%}")
            
            # Environmental pressure
            if pollution_level < 0.3:
                pressure = "Favors Light"
            elif pollution_level > 0.7:
                pressure = "Favors Dark"
            else:
                pressure = "Neutral"
            
            pressure_text = pressure if gen > 0 else f"{pressure} (Not Applied)"
            st.markdown(f"**Environment:** {pressure_text}")

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
        st.metric("Final Total Population", total_pop_history[-1])
        st.metric("Population Growth", f"{((total_pop_history[-1] / initial_population - 1) * 100):+.1f}%")
    
    with final_col2:
        st.metric("Final White Population", white_history[-1])
        st.metric("Final Dark Population", dark_history[-1])
    
    with final_col3:
        initial_white_pct = (initial_white / initial_population * 100)
        final_white_pct = (white_history[-1] / total_pop_history[-1] * 100) if total_pop_history[-1] > 0 else 0
        change_white = final_white_pct - initial_white_pct
        
        st.metric("White Moth %", f"{final_white_pct:.1f}%", f"{change_white:+.1f}%")
        st.metric("Dark Moth %", f"{100-final_white_pct:.1f}%", f"{-change_white:+.1f}%")

    # Educational conclusion
    st.markdown("---")
    st.subheader("Scientific Insights")
    
    population_growth_insight = ""
    if total_pop_history[-1] > initial_population * 1.5:
        population_growth_insight = " The population grew significantly, showing successful adaptation."
    elif total_pop_history[-1] < initial_population * 0.8:
        population_growth_insight = " Population declined due to environmental pressure."
    else:
        population_growth_insight = " Population remained relatively stable."
    
    if final_white_pct < 25:
        conclusion = f"The dark moths became dominant due to increased pollution.{population_growth_insight}"
    elif final_white_pct > 75:
        conclusion = f"The white moths remained dominant in the cleaner environment.{population_growth_insight}"
    else:
        conclusion = f"Both variants coexisted, showing balanced selection pressure.{population_growth_insight}"
    
    st.info(f"**Conclusion:** {conclusion}")
    
    st.markdown("**Note:** Generation 1 shows the baseline population before natural selection begins.")

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
        - Population size affects moth density
        - Generation 1 shows baseline (no selection)
        """)
    
    with col2:
        st.markdown("""
        **Population Tracking**
        - Real-time population graphs showing moth variants
        - Environmental change over time
        - Statistical summaries and metrics
        - Clear indication when selection pressure begins
        """)